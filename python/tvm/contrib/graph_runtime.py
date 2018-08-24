"""Minimum graph runtime that executes graph containing TVM PackedFunc."""
from .._ffi.base import string_types
from .._ffi.function import get_global_func
from ..rpc import base as rpc_base
from .. import ndarray as nd
import json
import re

BAR_LEN = 125 # Length of Bar that separates the arg attribute for profiles in json format. 
              # Makes output of chrome tracing look nicer.

def create(graph_json_str, libmod, ctx):
    """Create a runtime executor module given a graph and module.
    Parameters
    ----------
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.
    libmod : tvm.Module
        The module of the corresponding function
    ctx : TVMContext
        The context to deploy the module, can be local or remote.
    Returns
    -------
    graph_module : GraphModule
        Runtime graph module that can be used to execute the graph.
    """
    if not isinstance(graph_json_str, string_types):
        try:
            graph_json_str = graph_json_str._tvm_graph_json()
        except AttributeError:
            raise ValueError("Type %s is not supported" % type(graph_json_str))
    device_type = ctx.device_type
    device_id = ctx.device_id
    if device_type >= rpc_base.RPC_SESS_MASK:
        assert libmod.type_key == "rpc"
        assert rpc_base._SessTableIndex(libmod) == ctx._rpc_sess._tbl_index
        hmod = rpc_base._ModuleHandle(libmod)
        fcreate = ctx._rpc_sess.get_function("tvm.graph_runtime.remote_create")
        device_type = device_type % rpc_base.RPC_SESS_MASK
        return GraphModule(fcreate(graph_json_str, hmod, device_type, device_id), ctx)
    fcreate = get_global_func("tvm.graph_runtime.create")
    return GraphModule(fcreate(graph_json_str, libmod, device_type, device_id), ctx)


class GraphModule(object):
    """Wrapper runtime module.
    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions
    Parameters
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.
    ctx : TVMContext
        The context this module is under
    Attributes
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.
    ctx : TVMContext
        The context this module is under
    """
    def __init__(self, module, ctx):
        self.module = module
        self._set_input = module["set_input"]
        self._run = module["run"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        try:
            self._debug_get_output = module["debug_get_output"]
        except AttributeError:
            pass
        self._load_params = module["load_params"]
        self.ctx = ctx

        # profiling tool
        self.profile_data = False
        self._run_profile = module["run_profile"]
        self._get_op_start_time = module["get_op_start_time"]
        self._get_op_end_time = module["get_op_end_time"]
        self._get_op_size = module["get_op_size"]

    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module via kwargs
        Parameters
        ----------
        key : int or str
           The input key
        value : the input value.
           The input key
        params : dict of str to NDArray
           Additonal arguments
        """
        if key:
            self._set_input(key, nd.array(value, ctx=self.ctx))
        for k, v in params.items():
            self._set_input(k, nd.array(v, ctx=self.ctx))
        return self

    def run(self, profile = False, **input_dict):
        """Run forward execution of the graph
        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """

        if input_dict:
            self.set_input(**input_dict)

        if profile:
            self.profile_data = True
            self._run_profile()
        else:
            self._run()

    def get_input(self, index, out):
        """Get index-th input to out
        Parameters
        ----------
        index : int
            The input index
        out : NDArray
            The output array container
        """
        self._get_input(index, out)
        return out

    def get_output(self, index, out):
        """Get index-th output to out
        Parameters
        ----------
        index : int
            The input index
        out : NDArray
            The output array container
        """
        self._get_output(index, out)
        return out

    def debug_get_output(self, node, out):
        """Run graph upto node and get the output to out
        Parameters
        ----------
        node : int / str
            The node index or name
        out : NDArray
            The output array container
        """
        if hasattr(self, '_debug_get_output'):
            self._debug_get_output(node, out)
        else:
            raise RuntimeError("Please compile runtime with USE_GRAPH_RUNTIME_DEBUG = 0")
        return out

    def load_params(self, params_bytes):
        """Load parameters from serialized byte array of parameter dict.
        Parameters
        ----------
        params_bytes : bytearray
            The serialized parameter dict.
        """
        self._load_params(bytearray(params_bytes))

    def __getitem__(self, key):
        """Get internal module function
        Parameters
        ----------
        key : str
            The key to the module.
        """
        return self.module[key]

    def fetch_op(self,graph):

        if not self.profile_data:
            raise ValueError("Run module with profile first.")

        op_shape = graph.json_attr("shape")
        op_dtype = graph.json_attr("dltype")
        op_nodes = graph.index.nodes
        n_rp = graph.index.entry_ptr

        op_start = []
        for i in range(graph.index.num_nodes):
            op_start.append(self._get_op_start_time(i))

        op_end = []
        for i in range(graph.index.num_nodes):
            op_end.append(self._get_op_end_time(i))

        op_size = []
        for i in range(graph.index.num_nodes):
            op_size.append(self._get_op_size(i))

        return op_shape, op_dtype, op_nodes, n_rp, op_start, op_end, op_size

    def get_profile_data(self, graph):

        # get operation info
        op_shape, op_dtype, op_nodes, n_rp, op_start, op_end, op_size = self.fetch_op(graph)

        # create profile
        op_profile = []
        for i in range(graph.index.num_nodes):
            # skip non-operation nodes
            if op_nodes[i]["op"] == "null": continue
            # add operation information
            op_dict = {"OpName": op_nodes[i]["name"],
                       "OpType": op_nodes[i]["op"],
                       "StartTime": op_start[i],
                       "EndTime": op_end[i],
                       "Duration": op_end[i]-op_start[i],
                       "FuncName": op_nodes[i]["attrs"]["func_name"],
                       "NumInput": int(op_nodes[i]["attrs"]["num_inputs"]),
                       "NumOutput": int(op_nodes[i]["attrs"]["num_outputs"])}
            # add input information
            inputs = []
            for idx in range(int(op_nodes[i]["attrs"]["num_inputs"])):
                n_entry = op_nodes[i]["inputs"][idx]
                n_idx = n_rp[n_entry[0]]+n_entry[1]
                input_attr = {"InName": op_nodes[n_idx]["name"],
                              "InType": op_nodes[n_idx]["op"],
                              "InFunc": op_nodes[i]["attrs"]["func_name"],
                              "InDim": len(op_shape[n_idx]),
                              "InShape": op_shape[n_idx],
                              "InDType": op_dtype[n_idx],
                              "InSize": op_size[n_idx]}
                inputs.append({"Input_%d"%(idx+1):input_attr})
            op_dict["Input"] = inputs
            # add output information
            outputs = []
            for idx in range(int(op_nodes[i]["attrs"]["num_outputs"])):
                n_idx = n_rp[i]+idx
                output_attr = {"OutName": op_nodes[n_idx]["name"],
                               "OutType": op_nodes[n_idx]["op"],
                               "OutFunc": op_nodes[i]["attrs"]["func_name"],
                               "OutDim": len(op_shape[n_idx]),
                               "OutShape": op_shape[n_idx],
                               "OutDType": op_dtype[n_idx],
                               "OutSize": op_size[n_idx]}
                outputs.append({"Output_%d"%(idx+1):output_attr})
            op_dict["Output"] = outputs
            # update profile
            op_profile.append(op_dict)

        return op_profile

    def get_profile_json(self, graph, process=""):

        # get operation info
        op_shape, op_dtype, op_nodes, n_rp, op_start, op_end, op_size = self.fetch_op(graph)

        # create profile
        json_profile = []
        for i in range(graph.index.num_nodes):
            # skip non-operation nodes
            if op_nodes[i]["op"] == "null": continue
            # add operation information
            arg_dict = {"": "_"*BAR_LEN,
                        "Operation Name:": op_nodes[i]["name"],
                        "Operation Type:": op_nodes[i]["op"],
                        "Start Time (ms):": op_start[i],
                        "End Time (ms):": op_end[i],
                        "Duration (ms):": op_end[i]-op_start[i],
                        "Function Name:": op_nodes[i]["attrs"]["func_name"],
                        "Number of Inputs:": int(op_nodes[i]["attrs"]["num_inputs"]),
                        "Number of Outputs:": int(op_nodes[i]["attrs"]["num_outputs"])}
            # add input information
            inputs = []
            for idx in range(int(op_nodes[i]["attrs"]["num_inputs"])):
                n_entry = op_nodes[i]["inputs"][idx]
                n_idx = n_rp[n_entry[0]]+n_entry[1]
                input_attr = {"Input Name": op_nodes[n_idx]["name"],
                              "Input Type": op_nodes[n_idx]["op"],
                              "Input Function": op_nodes[i]["attrs"]["func_name"],
                              "Input Dimensions": len(op_shape[n_idx]),
                              "Input Shape": " x ".join(list(map(str,op_shape[n_idx]))),
                              "Input Data Type": op_dtype[n_idx],
                              "Input Size (bytes)": op_size[n_idx]}
                inputs.append({"Input_%d"%(idx+1):input_attr})
            arg_dict["Input"] = inputs
            # add output information
            outputs = []
            for idx in range(int(op_nodes[i]["attrs"]["num_outputs"])):
                n_idx = n_rp[i]+idx
                output_attr = {"Output Name": op_nodes[n_idx]["name"],
                               "Output Type": op_nodes[n_idx]["op"],
                               "Output Function": op_nodes[i]["attrs"]["func_name"],
                               "Output Dimensions": len(op_shape[n_idx]),
                               "Output Shape": " x ".join(list(map(str,op_shape[n_idx]))),
                               "Output Data Type": op_dtype[n_idx],
                               "Output Size (bytes)": op_size[n_idx]}
                outputs.append({"Output_%d"%(idx+1):output_attr})
            arg_dict["Output"] = outputs
            # update profile
            op = re.sub(r'\d+_','_',op_nodes[i]["attrs"]["func_name"])
            op = re.sub(r'_\d+$','',op)
            op = re.sub(r'\d+$','',op)
            json_profile.append({"name": op,
                                 "cat":  "DLC",
                                 "ph":   "B",
                                 "ts":   op_start[i]*1000,
                                 "pid":  process,
                                 "tid":  repr(self.ctx),
                                 "args": arg_dict})
            json_profile.append({"ph":   "E",
                                 "ts":   op_end[i]*1000,
                                 "pid":  process,
                                 "tid":  repr(self.ctx)})

        return json_profile


    def export_profile(self, fname, graph, process=""):
        with open(fname, "w") as of:
            json.dump(self.get_profile_json(graph, process), of, indent=2)