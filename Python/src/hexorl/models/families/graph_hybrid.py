"""Graph-hybrid crop model family descriptor."""

from hexorl.models.capabilities import (
    CROP_INPUT,
    DENSE_PLACE_POLICY,
    JOINT_PAIR_POLICY,
    REGRET_HEAD,
    SPARSE_PLACE_POLICY,
    CapabilitySet,
)
from hexorl.models.facets import make_descriptor
from hexorl.models.heads.pair_policy import CROP_PAIR_HEAD
from hexorl.models.heads.sparse_policy import GRAPH_HYBRID_POLICY_HEADS
from hexorl.models.heads.value import VALUE_HEAD
from hexorl.models.registry import FamilyComponents, ModelFamilyDescriptor
from hexorl.models.trunks.graph_hybrid import GRAPH_HYBRID_TRUNK, build_graph_hybrid_model

FAMILY_NAME = "graph_hybrid"
ALIASES = ("graph", "graph_hybrid_0")
CAPABILITIES = CapabilitySet.of(
    (CROP_INPUT, DENSE_PLACE_POLICY, SPARSE_PLACE_POLICY, JOINT_PAIR_POLICY, REGRET_HEAD)
)
COMPONENTS = FamilyComponents(
    trunk=GRAPH_HYBRID_TRUNK,
    heads=(*GRAPH_HYBRID_POLICY_HEADS, CROP_PAIR_HEAD, VALUE_HEAD),
)
REQUIRED_HEADS = ("policy", VALUE_HEAD)


def descriptor() -> ModelFamilyDescriptor:
    return make_descriptor(
        name=FAMILY_NAME,
        aliases=ALIASES,
        capabilities=CAPABILITIES,
        builder=build_graph_hybrid_model,
        components=COMPONENTS,
        required_heads=REQUIRED_HEADS,
        graph=False,
    )


__all__ = ["ALIASES", "CAPABILITIES", "COMPONENTS", "FAMILY_NAME", "REQUIRED_HEADS", "descriptor"]
