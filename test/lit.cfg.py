import os
import lit.formats
import lit.util

config.name = "TinyGPU-Compiler"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.tgc', '.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.tgc_obj_root, 'test')

config.substitutions.append(('%tgc', os.path.join(config.tgc_tools_dir, 'tgc')))

# Add tools directory to PATH
llvm_config = getattr(config, 'llvm_config', None)
if llvm_config:
    llvm_config.with_environment('PATH', config.tgc_tools_dir, append_path=True)
