import importlib.util
import os

replenish_path = os.path.abspath(os.path.dirname(__file__))
nodes_path = os.path.join(replenish_path, "nodes")

node_list = [
    "images_and_masks",
    "qq_nodes_option_load"
]


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def load_module_from_path(module_name, module_path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    except Exception as e:
        print(f"Failed to load module {module_name}: {e}")
    return None

for module_name in node_list:
    module_file_path = os.path.join(nodes_path, f"{module_name}.py")

    imported_module = load_module_from_path(module_name, module_file_path)
    
    if imported_module:
        NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
        if hasattr(imported_module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
