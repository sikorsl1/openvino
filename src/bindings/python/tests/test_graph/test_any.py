# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import OVAny
import pytest


@pytest.mark.parametrize(("value", "data_type"), [
    ("test_string", str),
    (2137, int),
    (21.37, float),
    (False, bool),
])
def test_any(value, data_type):
    ovany = OVAny(value)
    assert isinstance(ovany.value, data_type)
    assert ovany == value
    assert ovany.get() == value


@pytest.mark.parametrize(("values", "data_type"), [
    (["test", "string"], str),
    ([21, 37], int),
    ([21.0, 37.0], float),
])
def test_any_list(values, data_type):
    ovany = OVAny(values)
    assert isinstance(ovany.value, list)
    assert isinstance(ovany[0], data_type)
    assert isinstance(ovany[1], data_type)
    assert len(values) == 2
    assert ovany.get() == values


@pytest.mark.parametrize(("value_dict", "data_type"), [
    ({"key": "value"}, str),
    ({21: 37}, int),
    ({21.0: 37.0}, float),
])
def test_any_dict(value_dict, data_type):
    ovany = OVAny(value_dict)
    key = list(value_dict.keys())[0]
    assert isinstance(ovany.value, dict)
    assert ovany[key] == list(value_dict.values())[0]
    assert len(ovany.value) == 1
    assert type(ovany.value[key]) == data_type
    assert type(list(value_dict.values())[0]) == data_type
    assert ovany.get() == value_dict


def test_any_set_new_value():
    value = OVAny(int(1))
    assert isinstance(value.value, int)
    value = OVAny("test")
    assert isinstance(value.value, str)
    assert value == "test"


def test_any_class():
    class TestClass:
        def __init__(self):
            self.text = "test"

    value = OVAny(TestClass())
    assert isinstance(value.value, TestClass)
    assert value.value.text == "test"
