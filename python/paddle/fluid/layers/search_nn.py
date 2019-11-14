# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
All layers just related to the neural network.
"""

from __future__ import print_function

from ..layer_helper import LayerHelper

__all__ = [
    'search_fc',
    'search_seq_fc',
    'search_grnn',
    'search_seq_arithmetic',
    ]


def search_fc(
        input,
        size,
        param_attr=None,
        bias_attr=None,
        act=None,
        is_test=False,
        name=None):
    """

    TODO:
    """
    helper = LayerHelper('search_fc', **locals())
    dtype = input.dtype
    input_shape = list(input.shape)
    assert len(input_shape) == 2
    w_shape = [size, input_shape[1]]
    w = helper.create_parameter(attr=param_attr, shape=w_shape, dtype=dtype, is_bias=False)
    b_shape = [size]
    b = helper.create_parameter(attr=bias_attr, shape=b_shape, dtype=dtype, is_bias=False)
    res = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='search_fc',
        inputs={
            'X': input,
            'W': w,
            'b': b,
        },
        outputs={"Out": res, },
        attrs={'out_size': size, }
    )

    return res


def search_seq_fc(
        input,
        size,
        param_attr=None,
        bias_attr=None,
        act=None,
        is_test=False,
        name=None):
    """

    TODO:
    """
    helper = LayerHelper('search_seq_fc', **locals())
    dtype = input.dtype
    input_shape = list(input.shape)
    assert len(input_shape) == 2
    w_shape = [size, input_shape[1]]
    w = helper.create_parameter(attr=param_attr, shape=w_shape, dtype=dtype, is_bias=False)
    input_dict = {'X': input, 'W': w,}
    has_bias = False
    if bias_attr is not None:
        b_shape = [size]
        b = helper.create_parameter(attr=bias_attr, shape=b_shape, dtype=dtype, is_bias=False)
        input_dict['b'] = b
        has_bias = True
    res = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='search_seq_fc',
        inputs=input_dict,
        outputs={"Out": res, },
        attrs={'out_size': size, 'has_bias': has_bias}
    )

    return res


def search_grnn(
        input,
        num_input,
        num_hidden,
        param_attr_in,
        param_attr_hidden,
        dtype='float32',
        is_test=False,
        name=None):
    """

    TODO:
    """

    helper = LayerHelper('search_grnn', **locals())

    input_shape = list(input.shape)
    assert len(input_shape) == 2 and input_shape[-1] == num_input

    _cap_h = num_hidden
    _cap_e = input_shape[-1]
    wi_shape = [3, _cap_h, _cap_e]
    wh_shape = [3, _cap_h, _cap_h]
    wi = helper.create_parameter(
        attr=param_attr_in, shape=wi_shape, dtype=dtype, is_bias=False)
    wh = helper.create_parameter(
        attr=param_attr_hidden, shape=wh_shape, dtype=dtype, is_bias=False)

    grnn_res = helper.create_variable_for_type_inference(dtype)
    grnn_buffer = helper.create_variable_for_type_inference(dtype)
    grnn_idx_sorted_by_width = helper.create_variable_for_type_inference(dtype)
    grnn_layout_input = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='search_grnn',
        inputs={
            'X': input,
            'Wi': wi,
            'Wh': wh,
        },
        outputs={"Out": grnn_res,
                 "tmp_buffer": grnn_buffer,
                 'idx_sorted_by_width': grnn_idx_sorted_by_width,
                 'layout_input': grnn_layout_input
                 },
        attrs={'num_input': num_input, 'num_hidden': num_hidden}
    )

    return grnn_res


def search_seq_arithmetic(
        input_x,
        input_y,
        op_type,
        name=None):
    """
    :param input_x:
    :param input_y:
    :param op_type:
    :param name:
    :return:
    """
    helper = LayerHelper('search_seq_arithmetic', **locals())
    dtype = input_x.dtype

    res = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='search_seq_arithmetic',
        inputs={
            'X': input_x,
            'Y': input_y,
        },
        outputs={"Out": res},
        attrs={'op_type': op_type}
    )

    return res