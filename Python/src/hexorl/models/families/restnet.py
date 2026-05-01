"""RestNet model family descriptor."""

from hexorl.models.capabilities import CROP_INPUT, DENSE_PLACE_POLICY, REGRET_HEAD, CapabilitySet
from hexorl.models.facets import make_descriptor
from hexorl.models.heads.policy import DENSE_POLICY_HEADS
from hexorl.models.heads.value import VALUE_HEAD
from hexorl.models.registry import FamilyComponents, ModelFamilyDescriptor
from hexorl.models.specs import RestNetParams
from hexorl.models.trunks.crop_xformer import RESTNET_TRUNK, build_restnet_model

FAMILY_NAME = "restnet"
ALIASES: tuple[str, ...] = ()
CAPABILITIES = CapabilitySet.of((CROP_INPUT, DENSE_PLACE_POLICY, REGRET_HEAD))
COMPONENTS = FamilyComponents(trunk=RESTNET_TRUNK, heads=(*DENSE_POLICY_HEADS, VALUE_HEAD, "opp_policy", "regret_rank", "regret_value", "axis", "axis_delta_norm", "moves_left", "lookahead_4", "lookahead_12", "lookahead_36"))
REQUIRED_HEADS = ("policy", VALUE_HEAD)


def descriptor() -> ModelFamilyDescriptor:
    return make_descriptor(
        name=FAMILY_NAME,
        aliases=ALIASES,
        capabilities=CAPABILITIES,
        builder=build_restnet_model,
        components=COMPONENTS,
        params_schema=RestNetParams,
        required_heads=REQUIRED_HEADS,
        graph=False,
    )


__all__ = ["ALIASES", "CAPABILITIES", "COMPONENTS", "FAMILY_NAME", "REQUIRED_HEADS", "descriptor"]
