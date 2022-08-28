# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
import tvm.testing
from tvm import te
from tvm.topi.cuda import stable_sort_by_key_thrust
from tvm.topi.cuda.scan import exclusive_scan, scan_thrust, schedule_scan, adjacent_diff_dim_zero
from tvm.topi.cuda.sort import sort_dim_zero_thrust
from tvm.contrib.thrust import can_use_thrust, can_use_rocthrust
import numpy as np


thrust_check_func = {"cuda": can_use_thrust, "rocm": can_use_rocthrust}


def test_stable_sort_by_key():
    size = 6
    keys = te.placeholder((size,), name="keys", dtype="int32")
    values = te.placeholder((size,), name="values", dtype="int32")

    keys_out, values_out = stable_sort_by_key_thrust(keys, values)

    for target in ["cuda", "rocm"]:
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            continue

        with tvm.target.Target(target + " -libs=thrust") as tgt:
            if not thrust_check_func[target](tgt, "tvm.contrib.thrust.stable_sort_by_key"):
                print("skip because thrust is not enabled...")
                return

            dev = tvm.device(target, 0)
            s = te.create_schedule([keys_out.op, values_out.op])
            f = tvm.build(s, [keys, values, keys_out, values_out], target)

            keys_np = np.array([1, 4, 2, 8, 2, 7], np.int32)
            values_np = np.random.randint(0, 10, size=(size,)).astype(np.int32)
            keys_np_out = np.zeros(keys_np.shape, np.int32)
            values_np_out = np.zeros(values_np.shape, np.int32)
            keys_in = tvm.nd.array(keys_np, dev)
            values_in = tvm.nd.array(values_np, dev)
            keys_out = tvm.nd.array(keys_np_out, dev)
            values_out = tvm.nd.array(values_np_out, dev)
            f(keys_in, values_in, keys_out, values_out)

            ref_keys_out = np.sort(keys_np)
            ref_values_out = np.array([values_np[i] for i in np.argsort(keys_np)])
            tvm.testing.assert_allclose(keys_out.numpy(), ref_keys_out, rtol=1e-5)
            tvm.testing.assert_allclose(values_out.numpy(), ref_values_out, rtol=1e-5)


def test_exclusive_scan():
    for target in ["cuda", "rocm"]:
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            continue

        with tvm.target.Target(target + " -libs=thrust") as tgt:
            if not thrust_check_func[target](tgt, "tvm.contrib.thrust.sum_scan"):
                print("skip because thrust is not enabled...")
                return

            for ishape in [(10,), (10, 10), (10, 10, 10)]:
                values = te.placeholder(ishape, name="values", dtype="int32")

                scan, reduction = exclusive_scan(values, return_reduction=True)
                s = schedule_scan([scan, reduction])

                dev = tvm.device(target, 0)
                f = tvm.build(s, [values, scan, reduction], target)

                values_np = np.random.randint(0, 10, size=ishape).astype(np.int32)
                values_np_out = np.zeros(values_np.shape, np.int32)

                if len(ishape) == 1:
                    reduction_shape = ()
                else:
                    reduction_shape = ishape[:-1]

                reduction_np_out = np.zeros(reduction_shape, np.int32)

                values_in = tvm.nd.array(values_np, dev)
                values_out = tvm.nd.array(values_np_out, dev)
                reduction_out = tvm.nd.array(reduction_np_out, dev)
                f(values_in, values_out, reduction_out)

                ref_values_out = np.cumsum(values_np, axis=-1, dtype="int32") - values_np
                tvm.testing.assert_allclose(values_out.numpy(), ref_values_out, rtol=1e-5)
                ref_reduction_out = np.sum(values_np, axis=-1)
                tvm.testing.assert_allclose(reduction_out.numpy(), ref_reduction_out, rtol=1e-5)


def test_inclusive_scan():
    out_dtype = "int64"

    for target in ["cuda", "rocm"]:
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            continue

        with tvm.target.Target(target + " -libs=thrust") as tgt:
            if not thrust_check_func[target](tgt, "tvm.contrib.thrust.sum_scan"):
                print("skip because thrust is not enabled...")
                return

            for ishape in [(10,), (10, 10)]:
                values = te.placeholder(ishape, name="values", dtype="int32")

                scan = scan_thrust(values, out_dtype, exclusive=False)
                s = tvm.te.create_schedule([scan.op])

                dev = tvm.device(target, 0)
                f = tvm.build(s, [values, scan], target)

                values_np = np.random.randint(0, 10, size=ishape).astype(np.int32)
                values_np_out = np.zeros(values_np.shape, out_dtype)
                values_in = tvm.nd.array(values_np, dev)
                values_out = tvm.nd.array(values_np_out, dev)
                f(values_in, values_out)

                ref_values_out = np.cumsum(values_np, axis=-1, dtype=out_dtype)
                tvm.testing.assert_allclose(values_out.numpy(), ref_values_out, rtol=1e-5)

def test_sort_dim_zero():
    """Sort along the 0 dimension of the array.
    """ 
    shape = (3, 3)    
    data = te.placeholder(shape, name="data", dtype="int32")

    #import pdb; pdb.set_trace()
    indices = sort_dim_zero_thrust(data)

    for target in ["cuda"]:
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            continue

        with tvm.target.Target(target + " -libs=thrust") as tgt:
            if not thrust_check_func[target](tgt, "tvm.contrib.thrust.sort_dim_zero"):
                print("skip because thrust is not enabled...")
                return

            dev = tvm.device(target, 0)
            s = te.create_schedule([indices.op])
            f = tvm.build(s, [data, indices], target)

            data_np = np.array([[4,4,3], [2,2,4], [1,3,2]], np.int32)            
            indices_np = np.zeros((data_np.shape[0],), np.int32)
            data_in = tvm.nd.array(data_np, dev)            
            indices_out = tvm.nd.array(indices_np, dev)
            f(data_in, indices_out)

            #import pdb; pdb.set_trace()
            ref_indices = np.lexsort((data_np[:,2], data_np[:,1], data_np[:,0]))
            #print(data_np)
            #print(ref_indices)
            #print(indices_out.numpy())
            
            tvm.testing.assert_allclose(indices_out.numpy(), ref_indices, rtol=1e-5)            

def test_adjacent_diff():
    """Compute the adjacent difference along the 0 dimension of the array.
    """ 
    shape = (3, 3)    
    data = te.placeholder(shape, name="data", dtype="int32")
    sorted_indices = te.placeholder((shape[0],), name="sorted_indices", dtype="int32")
    #import pdb; pdb.set_trace()

    adjacent_diff = adjacent_diff_dim_zero(data, sorted_indices)

    for target in ["cuda"]:
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            continue

        with tvm.target.Target(target + " -libs=thrust") as tgt:
            if not thrust_check_func[target](tgt, "tvm.contrib.thrust.adjacent_difference_dim_zero"):
                print("skip because thrust is not enabled...")
                return

            dev = tvm.device(target, 0)
            s = te.create_schedule([adjacent_diff.op])
            f = tvm.build(s, [data, sorted_indices, adjacent_diff], target)
            
            data_np = np.array([[4,4,3], [2,2,4], [1,3,2]], np.int32)
            sorted_indices_np = np.array([0,2,1], np.int32)
            adjacent_diff_np = np.zeros((data_np.shape[0],), np.int32)
            data_in = tvm.nd.array(data_np, dev)
            sorted_indices_in = tvm.nd.array(sorted_indices_np, dev)
            adjacent_diff_out = tvm.nd.array(adjacent_diff_np, dev)
            f(data_in, sorted_indices_in, adjacent_diff_out)

            #import pdb; pdb.set_trace()
            ref_adjacent_diff = np.zeros((data_np.shape[0],), np.int32)
            ref_adjacent_diff[0] = 0
            def cmp(a, b, n):
                for i in range(n):
                    if data_np[a][i] != data_np[b][i]:
                        return 1
                return 0
            for i in range(1, data_np.shape[0]):
                ref_adjacent_diff[i] = cmp(i - 1, i, data_np.shape[1])                
            #import pdb; pdb.set_trace()

            tvm.testing.assert_allclose(adjacent_diff_out.numpy(), ref_adjacent_diff, rtol=1e-5)  


if __name__ == "__main__":
    test_stable_sort_by_key()
    test_exclusive_scan()
    test_inclusive_scan()
    test_sort_dim_zero()
    #test_adjacent_diff(
