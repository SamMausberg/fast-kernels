from fast_kernels import native_available, native_build_info


def test_native_module_imports() -> None:
    info = native_build_info()
    assert "available" in info
    assert isinstance(native_available(), bool)
