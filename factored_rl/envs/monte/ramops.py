def bcd2int(bcd_string):
    # Convert a binary coded decimal string to an integer
    nibbles = [bcd_string[i:i + 4] for i in range(0, len(bcd_string), 4)]
    digits = [format(int(nib, 2), '01d') for nib in nibbles]
    return int(''.join(digits), 10)

def _getIndex(address):
    assert type(address) == str and len(address) == 2
    row, col = tuple(address)
    row = int(row, 16) - 8
    col = int(col, 16)
    return row * 16 + col

def getByte(ram, address):
    # Return the byte at the specified emulator RAM location
    idx = _getIndex(address)
    return ram[idx]

def getByteRange(ram, start, end):
    # Return the bytes in the emulator RAM from start through end-1
    idx1 = _getIndex(start)
    idx2 = _getIndex(end)
    return ram[idx1:idx2]

def setByte(env, address, value):
    # set the given address in the rame of the given environment
    idx = _getIndex(address)
    env.setRAM(idx, value)
