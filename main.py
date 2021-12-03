import os
import shutil
import tempfile
import paddle.fluid as fluid

from collections import OrderedDict

def _load_state(path):
    """
    记载paddlepaddle的参数
    :param path:
    :return:
    """
    if os.path.exists(path + '.pdopt'):
        # XXX another hack to ignore the optimizer state
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = fluid.io.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = fluid.io.load_program_state(path)
    return state


if __name__ == '__main__':
    import torch
    state = _load_state('.\weights\picodet_s_320_coco.pdparams')
    new_state = OrderedDict()
    for k, v in state.items():
        # if
        if not isinstance(v, dict):
            new_state[k] = torch.FloatTensor(v)
            # print(k, v.shape)
    torch.save(new_state, "./picodet_s_320_coco.pth")



