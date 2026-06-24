from .join_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .collector_nodes import VACEClipCollector
from .bridge_nodes import VACETwoVideoBridgeAssemble, VACETwoVideoBridgePrep

NODE_CLASS_MAPPINGS = dict(NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS["VACEClipCollector"] = VACEClipCollector
NODE_DISPLAY_NAME_MAPPINGS["VACEClipCollector"] = "VACE Clip Collector"
NODE_CLASS_MAPPINGS["VACETwoVideoBridgePrep"] = VACETwoVideoBridgePrep
NODE_CLASS_MAPPINGS["VACETwoVideoBridgeAssemble"] = VACETwoVideoBridgeAssemble
NODE_DISPLAY_NAME_MAPPINGS["VACETwoVideoBridgePrep"] = "VACE Two Video Bridge Prep"
NODE_DISPLAY_NAME_MAPPINGS["VACETwoVideoBridgeAssemble"] = "VACE Two Video Bridge Assemble"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
