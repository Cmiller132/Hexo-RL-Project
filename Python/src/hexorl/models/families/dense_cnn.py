"""Dense CNN model family descriptor."""

from hexorl.models.capabilities import CROP_INPUT, DENSE_PLACE_POLICY, REGRET_HEAD, CapabilitySet
from hexorl.models.facets import make_descriptor
from hexorl.models.heads.policy import DENSE_POLICY_HEADS
from hexorl.models.heads.value import VALUE_HEAD
from hexorl.models.registry import FamilyComponents, ModelFamilyDescriptor
from hexorl.models.trunks.dense_cnn import DENSE_CNN_TRUNK, build_dense_cnn_model

FAMILY_NAME = "dense_cnn"
ALIASES = ("cnn",)
CAPABILITIES = CapabilitySet.of((CROP_INPUT, DENSE_PLACE_POLICY, REGRET_HEAD))
COMPONENTS = FamilyComponents(trunk=DENSE_CNN_TRUNK, heads=(*DENSE_POLICY_HEADS, VALUE_HEAD))
REQUIRED_HEADS = ("policy", VALUE_HEAD)


def descriptor() -> ModelFamilyDescriptor:
    return make_descriptor(
        name=FAMILY_NAME,
        aliases=ALIASES,
        capabilities=CAPABILITIES,
        builder=build_dense_cnn_model,
        components=COMPONENTS,
        required_heads=REQUIRED_HEADS,
        graph=False,
    )


__all__ = ["ALIASES", "CAPABILITIES", "COMPONENTS", "FAMILY_NAME", "REQUIRED_HEADS", "descriptor"]
