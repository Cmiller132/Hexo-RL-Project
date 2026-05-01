"""Global cross-attention model family descriptor."""

from hexorl.models.capabilities import (
    GLOBAL_GRAPH_INPUT,
    GLOBAL_PLACE_POLICY,
    JOINT_PAIR_POLICY,
    PAIR_FIRST_POLICY,
    PAIR_SECOND_POLICY,
    REGRET_HEAD,
    CapabilitySet,
)
from hexorl.models.facets import make_descriptor
from hexorl.models.heads.tactical import GLOBAL_GRAPH_OUTPUT_HEADS
from hexorl.models.registry import FamilyComponents, ModelFamilyDescriptor
from hexorl.models.specs import GlobalGraphParams
from hexorl.models.trunks.global_xattn import GLOBAL_XATTN_TRUNK, build_global_xattn_model

FAMILY_NAME = "global_xattn"
ALIASES = ("global_xattn_0",)
CAPABILITIES = CapabilitySet.of(
    (GLOBAL_GRAPH_INPUT, GLOBAL_PLACE_POLICY, PAIR_FIRST_POLICY, PAIR_SECOND_POLICY, JOINT_PAIR_POLICY, REGRET_HEAD)
)
COMPONENTS = FamilyComponents(trunk=GLOBAL_XATTN_TRUNK, heads=(*GLOBAL_GRAPH_OUTPUT_HEADS, "opp_policy", "regret_rank", "regret_value", "moves_left", "tactical", "legal_token_quality", "lookahead_4", "lookahead_12", "lookahead_36"))
REQUIRED_HEADS = ("value", "policy_place")


def descriptor() -> ModelFamilyDescriptor:
    return make_descriptor(
        name=FAMILY_NAME,
        aliases=ALIASES,
        capabilities=CAPABILITIES,
        builder=build_global_xattn_model,
        components=COMPONENTS,
        params_schema=GlobalGraphParams,
        required_heads=REQUIRED_HEADS,
        graph=True,
    )


__all__ = ["ALIASES", "CAPABILITIES", "COMPONENTS", "FAMILY_NAME", "REQUIRED_HEADS", "descriptor"]
