# How to test OpenVINO™ Python API?

#### Building and environment
Instructions can be found in ["Building the OpenVINO™ Python API"](./build.md).

### Running OpenVINO™ Python API tests
*For simplicity, all of these commands require to navigate to the main Python API folder first:*
```shell
cd .../openvino/src/bindings/python/
```

To run **all tests**:
```shell
pytest tests/
```

Test framework *pytest* allows to filter tests with `-k` flag.
```shell
pytest tests/test_runtime/test_core.py -k "test_available_devices"
```

Alternatively, the full name and path to the test case could be passed.
```shell
pytest tests/test_runtime/test_core.py::test_available_devices
```

To print test names and increase verbosity, use `-v` flag.
```shell
pytest tests/test_runtime/test_core.py -v
```
*Tip: look at pytest's documentation for more useful tricks: https://docs.pytest.org/en/latest/*

To run full test suite one can utilize `tox` command:
```shell
tox
```

### Writing OpenVINO™ Python API tests
###### Before start
Follow and complete `openvino/src/bindings/python/docs/code_examples.md`.

##### Adding new test-case in the correct place
Let's add a new test for OpenVINO™ Python API.

First, the test should confirm that the new pybind11-based class of `MyTensor` is behaving correctly. Navigate to tests folder and create a new file that describes tests within it. It should be along the lines of:

    tests/test_runtime/test_mytensor.py


**Don't forget to include license on the top of each new file!**

Note that name of the file is connected to the class/module to be tested. This is exactly why tests are structured in folders that are describing what tests are supposed to be there. Always add tests to correct places, new folders and files should be created only when necessary. Quick overview of the structure:

    tests/test_frontend           <-- frontend manager and extensions
    tests/test_runtime            <-- runtime classes such as Core and Tensor
    tests/test_graph              <-- operators and their implementation
    tests/test_onnx               <-- ONNX Frontend tests and validation
    tests/test_transformations    <-- optimization passes for OV Models 

##### Writing test itself
Let's add a test case for new class. Start with imports and simple test of the creation of a class:
```python
import pytest
import numpy as np 
import openvino.runtime as ov

def test_mytensor_creation():
    tensor = ov.MyTensor([1, 2, 3])

    assert tensor is not None
```

Rebuilding step is not necessary here as long as there are no updates to codebase itself. Run the test with:
```shell
pytest tests/test_runtime/test_mytensor.py -v
```

In actual tests it is a good pratice to parametrize them, thus making tests compact and reducing number of handwritten test cases. Additionally, adding checks for shared functions to the basic tests is a common technique. Let's replace the test with:
```python
@pytest.mark.parametrize(("source"), [
    ([1, 2, 3]),
    (ov.Tensor(np.array([4, 5 ,6]).astype(np.float32))),
])
def test_mytensor_creation(source):
    tensor = ov.MyTensor(source)

    assert tensor is not None
    assert tensor.get_size() == 3
```

Run the tests, output should be similar to:
```shell
tests/test_runtime/test_mytensor.py::test_mytensor_creation[source0] PASSED                                                                                                                                    [ 50%]
tests/test_runtime/test_mytensor.py::test_mytensor_creation[source1] PASSED                                                                                                                                    [100%]
```

Notice that the test name is shared between cases. In a real-life pull request, all of the functionalities should be tested to ensure the quality of the solution. Always focus on general usage and edge-case scenarios. On the other hand, remember that excessive testing is not advised as it may result in duplicate test cases and impact validation pipelines. A good "rule-of-thumb" list of practices while adding tests to the project is:
* Don't test built-in capabilities of a given language.
* Common functions can be tested together.
* Create test cases with a few standard scenarios and cover all known edge-cases.  
* Hardcode desired results...
* ... or create reference values during runtime. Always use a good, thrust-worthy library for that!
* Re-use common parts of the code (like multiple lines that create helper object) and move them out to make tests easier to read.

###### Difference between *tests* and *tests_compatibility* directories
<!-- TO-DELETE when compatibility layer is no longer supported in the project -->
Someone could notice two similar folders `tests` and `tests_compatibility`. First one is the desired place for all upcoming features and tests. Compatibility layer is only supported in specific cases and any updates to it should be explicitly approved by OpenVINO™ reviewers. Please do not duplicate tests in both directories if not necessary.
