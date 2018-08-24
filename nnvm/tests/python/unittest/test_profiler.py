import os
import tvm
import nnvm
import mxnet as mx
import numpy as np
from mxnet.gluon.model_zoo.vision import get_model
from tvm.contrib import graph_runtime
from nnvm.frontend import from_mxnet

def get_profiled_model():
    model_name = "resnet50_v1"
    dshape = (1,3,224,224)
    ctx = tvm.cpu()
    target = "llvm"

    block = get_model(model_name , pretrained=True)
    img = np.ones(dshape).astype(np.float32)

    net, params = from_mxnet(block)
    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(
            net, target, shape={"data": dshape}, params=params)
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("data", tvm.nd.array(img, ctx=ctx))
    module.run(profile=True)

    return module, graph

def test_get_profile_data():

    module, graph = get_profiled_model()
    profile = module.get_profile_data(graph)

def test_get_profile_json():

    module, graph = get_profiled_model()
    profile = module.get_profile_json(graph)

def test_export_profile():
    fname = "./test_profiler.json"
    module, graph = get_profiled_model()
    module.export_profile(fname, graph)
    if os.path.isfile(fname):
        os.remove(fname)

if __name__ == "__main__":
    test_get_profile_data()
    test_get_profile_json()
    test_export_profile()