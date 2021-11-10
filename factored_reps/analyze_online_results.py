def analyze_results(output_dir, replay_test, fnet, predictor):
    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    fields = ['_index_', 'ob', 'state', 'action', 'next_ob']
    test_batch = replay_test.retrieve(fields=fields)
    idx, obs, states, actions, next_obs = test_batch

    def compute_accuracy(labels, predictions):
        n_correct = (labels == predictions).sum()
        n_total = len(labels)
        accuracy = 100 * n_correct / n_total
        return n_correct, n_total, accuracy

    def record_model_accuracy(output_file, description, n_correct, n_total, accuracy):
        output_file.write('{}: {} correct out of {} total = {}%\n'.format(
            description, n_correct, n_total, accuracy))
        output_file.write('\n')

    with open(os.path.join(output_dir, 'results.txt'), 'w') as output_file:
        #% ------------------ Compute inverse model accuracy ------------------
        predicted_actions = fnet.predict_a(obs, next_obs).detach().cpu().numpy()
        inv_results = compute_accuracy(actions.detach().cpu().numpy(), predicted_actions)
        record_model_accuracy(output_file, 'Inverse model accuracy', *inv_results)

        #% ------------------ Compute discriminator accuracy ------------------
        predicted_is_fake_on_positives = fnet.predict_is_fake(obs, next_obs).detach().cpu().numpy()
        discrim_results_positives = compute_accuracy(np.zeros_like(predicted_is_fake_on_positives),
                                                     predicted_is_fake_on_positives)
        record_model_accuracy(output_file, 'Discriminator accuracy (positives)',
                              *discrim_results_positives)

        for mode in ['random', 'same', 'following']:
            negatives = fnet.get_negatives(replay_test, idx, mode=mode)
            predicted_is_fake_on_negatives = fnet.predict_is_fake(
                obs, negatives).detach().cpu().numpy()
            discrim_results_negatives = compute_accuracy(
                np.ones_like(predicted_is_fake_on_negatives), predicted_is_fake_on_negatives)
            record_model_accuracy(output_file,
                                  'Discriminator accuracy ({}-state negatives)'.format(mode),
                                  *discrim_results_negatives)

    #% ------------------ Generate predictor confusion plots ------------------
    z0 = fnet.encode(obs)
    test_reconstructions = predictor.predict(z0).detach().cpu().numpy()

    def generate_confusion_plots(s_actual, s_predicted):
        state_vars = ['taxi_row', 'taxi_col', 'passenger_row', 'passenger_col',
                      'in_taxi'][:len(states[0])]

        fig, axes = plt.subplots(len(state_vars), 1, figsize=(3, 2 * len(state_vars)))

        for (state_var_idx, state_var), ax in zip(enumerate(state_vars), axes):
            bins = len(np.unique(s_actual[:, state_var_idx]))
            h = ax.hist2d(x=s_predicted[:, state_var_idx], y=s_actual[:, state_var_idx], bins=bins)
            counts, xedges, yedges, im = h
            fig.colorbar(im, ax=ax)
            # sns.histplot(
            #     x=s_predicted[:, state_var_idx],
            #     y=s_actual[:, state_var_idx],
            #     bins=bins,
            #     discrete=True,
            #     cbar=True,
            #     stat='count',
            #     ax=ax,
            # )
            print(counts)
            ax.set_title(state_var)
            ax.set_xlabel('predicted')
            ax.set_ylabel('actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predictor_confusion_plots.png'),
                    facecolor='white',
                    edgecolor='white')
        # plt.show()

    generate_confusion_plots(states.detach().cpu().numpy(), test_reconstructions)
