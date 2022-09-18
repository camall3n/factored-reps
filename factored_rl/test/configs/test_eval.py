import pytest
import hydra

from omegaconf import OmegaConf
from omegaconf.errors import GrammarParseError
from yaml.scanner import ScannerError

OmegaConf.register_new_resolver("eval", eval)

# Test A: Quote the entire RHS, with a space after `eval:`
def test_1a_eval_outer_with_space():
    cfg = OmegaConf.create("""
    a: '${eval: 2 * 3 * 5}'
    """)
    assert cfg.a == 30 # => pass!

def test_2a_nested_outer_with_space():
    with pytest.raises(GrammarParseError):
        cfg = OmegaConf.create("""
        foo: bar
        baz: '${eval: 1 if "${foo}" == "bar" else 0}'
        """) #        => GrammarParseError: token recognition error at: '='
        assert cfg.baz == 1

# --------------------------------------------------------------------
# Test B: Quote the insides only, with a space after `eval:`
def test_1b_eval_inner_with_space():
    with pytest.raises(ScannerError):
        cfg = OmegaConf.create("""
        a: ${eval: '2 * 3 * 5'}
        """) #   ^    => ScannerError: mapping values are not allowed here
        assert cfg.a == 30

def test_2b_nested_inner_with_space():
    with pytest.raises(ScannerError):
        cfg = OmegaConf.create("""
        foo: bar
        baz: ${eval: '1 if "${foo}" == "bar" else 0'}
        """) #     ^  => ScannerError: mapping values are not allowed here
        assert cfg.baz == 1

# --------------------------------------------------------------------
# Test C: Quote entire RHS, with no space after `eval:`
def test_1c_eval_outer_no_space():
    cfg = OmegaConf.create("""
    a: '${eval:2 * 3 * 5}'
    """)
    assert cfg.a == 30 # => pass!

def test_2c_nested_outer_no_space():
    with pytest.raises(GrammarParseError):
        cfg = OmegaConf.create("""
        foo: bar
        baz: '${eval:1 if "${foo}" == "bar" else 0}'
        """) #        => GrammarParseError: mismatched input '"' expecting BRACE_CLOSE
        assert cfg.baz == 1 # => pass!

# --------------------------------------------------------------------
# Test D: Quote the insides only, with no space after `eval:`
def test_1d_eval_inner_no_space():
    cfg = OmegaConf.create("""
    a: ${eval:'2 * 3 * 5'}
    """)
    assert cfg.a == 30 # => pass!

def test_2d_nested_inner_no_space():
    cfg = OmegaConf.create("""
    foo: bar
    baz: ${eval:'1 if "${foo}" == "bar" else 0'}
    """)
    assert cfg.baz == 1 # => pass!

# --------------------------------------------------------------------

def test_eval_three():
    cfg = OmegaConf.create('three: ${eval:"1 + 2"}')
    assert cfg.three == 3

def test_eval_hydra():
    with hydra.initialize(version_base=None, config_path='./conf'):
        cfg = hydra.compose(config_name='eval')

    assert cfg.name == 'eval'
    assert cfg.one == 1
    assert cfg.two == 2
    assert cfg.three == 3

def test_eval_hydra_overrides():
    with hydra.initialize(version_base=None, config_path='./conf'):
        cfg = hydra.compose(config_name='eval', overrides=["two=4"])

    assert cfg.name == 'eval'
    assert cfg.one == 1
    assert cfg.two == 4
    assert cfg.three == 5

def test_nested_interpolations():
    with hydra.initialize(version_base=None, config_path='./conf'):
        cfg = hydra.compose(config_name='nested')

    assert cfg.name == 'foo'
    assert cfg.value == 1

    with hydra.initialize(version_base=None, config_path='./conf'):
        cfg = hydra.compose(config_name='nested', overrides=["name=bar"])

    assert cfg.name == 'bar'
    assert cfg.value == 2
