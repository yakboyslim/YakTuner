import numpy as np
def bin_read(bin_file, req):
    """
    Read binary data according to specified requirements
    """
    output = []

    if isinstance(req[0], int):
        bin_file.seek(req[0])
        temp = np.fromfile(bin_file, dtype=req[5], count=req[1] * req[2])
        temp = temp.reshape((req[1], req[2]),order='F')
        temp = temp.astype(int)
        temp = temp - req[3]
        temp = temp / req[4]
        output.append(temp)
    else:
        for j in range(len(req[0])):
            bin_file.seek(req[0][j])
            temp = np.fromfile(bin_file, dtype=req[5][j], count=req[1][j] * req[2][j])
            temp = temp.reshape((req[1][j], req[2][j]),order='F')
            temp = temp.astype(int)
            temp = temp - req[3][j]
            temp = temp / req[4][j]
            output.append(temp)

    return output
