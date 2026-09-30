"""Semantic strategy and mask-policy resolution.

This layer translates metadata into capabilities. It does not run models or
touch HTTP state, which keeps category behavior testable and centralized.
"""

import os
import re

from sam_runtime import *


def normalize_sam_quality_profile(profile):
    value = str(profile or "").strip().lower()
    if value in {"completion", "completion_candidate", "completion_review", "canonical_completion"}:
        return "completion"
    return "publish"


def is_price_like_text(text):
    value = str(text or "").lower()
    return any(token in value for token in [
        "price_badge", "price", "badge", "circle", "round", "coin", "sticker", "$",
        "价格", "价签", "徽章"
    ])


def is_completion_segmentation_layer(layer_meta):
    """Identify post-inpaint SAM requests without relaxing first-pass SAM."""
    meta = layer_meta or {}
    return bool(
        meta.get("completionSegmentation") or
        str(meta.get("id") or "").startswith("completion-scene-sam-") or
        str(meta.get("layerId") or "").startswith("completion-scene-sam-")
    )

def _get_legacy_layer_strategy(layer_meta):
    extraction_profile = str(layer_meta.get("extractionProfile", "")).lower()
    semantic_type = str(layer_meta.get("semanticType", "")).lower()
    design_role = str(layer_meta.get("designRole", "")).lower()
    profile_text = " ".join([
        str(layer_meta.get("name", "")),
        str(layer_meta.get("semanticType", "")),
        str(layer_meta.get("designRole", "")),
        str(layer_meta.get("category", "")),
        str(layer_meta.get("runtimeType", ""))
    ]).lower()

    intent = layer_meta.get("segmentationIntent")
    intent_confidence = 0.0
    if isinstance(intent, dict):
        try:
            intent_confidence = float(intent.get("confidence") or 0.0)
        except (TypeError, ValueError):
            intent_confidence = 0.0

    # A high-confidence tight atomic-object contract is stronger than the
    # broad upstream food/product label. Keep these hard silhouettes on the
    # shared hard-product candidate and matte route; the generic semantic
    # identity checks still prevent adjacent layers from being absorbed.
    if (
        extraction_profile == "layout_embedded_product" and
        isinstance(intent, dict) and
        str(intent.get("targetKind") or "").lower() == "atomic_object" and
        str(intent.get("bboxRole") or "").lower() == "tight_subject" and
        intent_confidence >= 0.90 and
        semantic_type not in {"person", "person_character"}
    ):
        return {
            "type": "hard_product",
            "max_fill": 0.88,
            "max_merged_fill": 0.86,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 6,
            "require_overlap_for_attachments": False
        }

    # Plant layers commonly contain narrow leaves/fronds rather than one
    # filled silhouette. Keep their low-fill SAM parts eligible for the
    # established hard-product attachment selector.
    if semantic_type == "decor_plant" and design_role == "scene_object":
        return {
            "type": "hard_product",
            "max_fill": 0.88,
            "max_merged_fill": 0.86,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 6,
            "require_overlap_for_attachments": False
        }

    if any(token in str(layer_meta.get("name", "")).lower() for token in [
        "羽毛", "羽绒", "羽饰", "窗纱", "薄纱", "纱帘", "透明织物"
    ]):
        return {
            "type": "soft_edge",
            "max_fill": 0.96,
            "max_merged_fill": 0.96,
            "max_attachment_distance": 24,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 8,
            "require_overlap_for_attachments": False
        }

    # layout_embedded_product is a broad upstream profile used for poster
    # subjects, but it can also be attached to a physical surface by the
    # semantic layer pass. Resolve explicit hard-surface signals first so a
    # table/console cannot be routed through the food-specific pipeline.
    explicit_food_layer = (
        semantic_type in {"product_food", "product_drink"} or
        design_role == "product_image"
    )
    embedded_table_signal = any(token in profile_text for token in [
        "table", "desk", "console", "sideboard", "cabinet", "counter",
        "茶几", "桌面", "台面", "边几", "矮几", "桌", "柜"
    ])
    if (
        extraction_profile == "layout_embedded_product" and
        embedded_table_signal and
        not explicit_food_layer
    ):
        return {
            "type": "table",
            "max_fill": 0.66,
            "max_merged_fill": 0.72,
            "max_attachment_distance": 7,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 3,
            "require_overlap_for_attachments": True
        }

    if extraction_profile in ["vector_layout_element", "text_layer", "background_plate"]:
        return {
            "type": "flat_shape",
            "max_fill": 0.92,
            "max_merged_fill": 0.82,
            "max_attachment_distance": 6,
            "allow_attachments": False,
            "prefer_rectangular": extraction_profile != "text_layer",
            "max_masks": 1,
            "require_overlap_for_attachments": True
        }

    # Upstream semantic labels are sometimes coarse: a poster character can
    # arrive as product_food simply because the poster also advertises food.
    # An explicit person/character name is stronger evidence than that broad
    # label. Route it as one hard entity before layout_embedded_product would
    # otherwise send it through compound-food expansion and cleanup.
    person_profile_text = re.sub(r"\bsemantic_object\b", "", profile_text)
    if any(token in person_profile_text for token in [
        "person", "people", "woman", "man", "girl", "boy", "portrait",
        "character", "figure", "人物", "女子", "女人", "女孩", "男孩",
        "男性", "女性", "肖像", "角色"
    ]):
        return {
            "type": "hard_product",
            "max_fill": 0.88,
            "max_merged_fill": 0.86,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 6,
            "require_overlap_for_attachments": False
        }

    if extraction_profile == "layout_embedded_product":
        return {
            "type": "food_product",
            "max_fill": 0.94,
            "max_merged_fill": 0.95,
            "max_attachment_distance": 16,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 5,
            "require_overlap_for_attachments": False
        }

    if extraction_profile == "multi_part_hard_product":
        return {
            "type": "hard_product",
            "max_fill": 0.88,
            "max_merged_fill": 0.86,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 6,
            "require_overlap_for_attachments": False
        }

    if extraction_profile == "compound_object":
        return {
            "type": "decor_arrangement",
            "max_fill": 0.92,
            "max_merged_fill": 0.90,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 8,
            "require_overlap_for_attachments": False
        }

    if extraction_profile == "multi_part_hard_object":
        if any(token in profile_text for token in [
            "lamp", "chandelier", "pendant", "lighting", "light fixture",
            "吊灯", "灯具", "灯饰", "照明", "吸顶灯", "壁灯"
        ]):
            return {
                "type": "lighting",
                "max_fill": 0.62,
                "max_merged_fill": 0.54,
                "max_attachment_distance": 6,
                "allow_attachments": True,
                "prefer_rectangular": False,
                "max_masks": 3,
                "require_overlap_for_attachments": True
            }
        if any(token in profile_text for token in [
            "table", "desk", "console", "sideboard", "cabinet", "台", "桌", "柜", "玄关"
        ]):
            return {
                "type": "table",
                "max_fill": 0.66,
                "max_merged_fill": 0.72,
                "max_attachment_distance": 7,
                "allow_attachments": True,
                "prefer_rectangular": False,
                "max_masks": 3,
                "require_overlap_for_attachments": True
            }
        return {
            "type": "furniture",
            "max_fill": 0.70,
            "max_merged_fill": 0.58,
            "max_attachment_distance": 7,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 3,
            "require_overlap_for_attachments": True
        }

    # An explicit hard-object profile takes precedence over descriptive words
    # such as "glass" or "transparent" in the layer name. Tables and cabinets
    # must not be routed through the soft-edge prompt/matte chain.
    if any(token in profile_text for token in [
        "feather", "plume", "hair", "fur", "smoke", "cloud", "fog",
        "sheer", "curtain", "tulle", "transparent fabric", "glass",
        "羽毛", "毛发", "绒毛", "烟", "云", "雾", "窗纱", "纱", "薄纱",
        "玻璃", "透明"
    ]):
        return {
            "type": "soft_edge",
            "max_fill": 0.96,
            "max_merged_fill": 0.96,
            "max_attachment_distance": 24,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 8,
            "require_overlap_for_attachments": False
        }

    text = " ".join([
        str(layer_meta.get("name", "")),
        str(layer_meta.get("semanticType", "")),
        str(layer_meta.get("category", "")),
        str(layer_meta.get("runtimeType", ""))
    ]).lower()

    if any(token in text for token in [
        "price_badge", "price", "badge", "价格", "价签", "徽章", "$"
    ]):
        return {
            "type": "flat_shape",
            "max_fill": 0.86,
            "max_merged_fill": 0.72,
            "max_attachment_distance": 6,
            "allow_attachments": False,
            "prefer_rectangular": False,
            "max_masks": 1,
            "require_overlap_for_attachments": True
        }

    if any(token in text for token in [
        "shape_panel", "ad_background", "cta_button", "logo_mark", "element_text",
        "text_node", "panel", "label", "card", "shape", "button", "logo",
        "面板", "底板", "色块", "背景框", "文字背景", "标签底板", "文字", "文本", "标志"
    ]):
        return {
            "type": "flat_shape",
            "max_fill": 0.92,
            "max_merged_fill": 0.82,
            "max_attachment_distance": 6,
            "allow_attachments": False,
            "prefer_rectangular": True,
            "max_masks": 1,
            "require_overlap_for_attachments": True
        }

    # Do not match the single character "画": names such as 插画人物 are
    # product/scene subjects, not wall art. Use explicit bounded-plane terms.
    if any(token in text for token in [
        "wall art", "painting", "artwork", "picture", "poster",
        "挂画", "装饰画", "壁画", "绘画作品", "墙面艺术"
    ]):
        return {
            "type": "wall_art",
            # A painting can legitimately occupy almost its entire semantic
            # bbox. High fill is not background evidence for a bounded plane.
            "max_fill": 0.98,
            "max_merged_fill": 0.98,
            "max_attachment_distance": 8,
            "allow_attachments": False,
            "prefer_rectangular": True,
            "max_masks": 1,
            "require_overlap_for_attachments": True
        }

    if any(token in text for token in [
        "lamp", "chandelier", "pendant", "lighting", "light fixture",
        "吊灯", "灯具", "灯饰", "照明", "吸顶灯", "壁灯"
    ]):
        return {
            "type": "lighting",
            "max_fill": 0.62,
            "max_merged_fill": 0.54,
            "max_attachment_distance": 6,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 3,
            "require_overlap_for_attachments": True
        }

    if any(token in text for token in [
        "feather", "plume", "hair", "fur", "smoke", "cloud", "fog",
        "sheer", "curtain", "tulle", "transparent fabric", "glass",
        "羽毛", "毛发", "绒毛", "烟", "云", "雾", "窗纱", "纱", "薄纱",
        "玻璃", "透明"
    ]):
        return {
            "type": "soft_edge",
            "max_fill": 0.96,
            "max_merged_fill": 0.96,
            "max_attachment_distance": 24,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 8,
            "require_overlap_for_attachments": False
        }

    if any(token in text for token in [
        "food", "dish", "meal", "plate", "rice", "fried rice", "pork", "roasted",
        "burger", "pizza", "noodle", "salad", "cola", "tea", "coffee", "choco",
        "drink", "beverage", "cup", "product_food", "product_drink",
        "食物", "食品", "菜品", "餐盘", "炒饭", "米饭", "猪肉", "烤肉", "饮料",
        "可乐", "茶", "咖啡", "热巧", "杯"
    ]):
        return {
            "type": "food_product",
            "max_fill": 0.94,
            "max_merged_fill": 0.95,
            "max_attachment_distance": 16,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 5,
            "require_overlap_for_attachments": False
        }

    if any(token in text for token in [
        "product_image", "product_packaging", "product", "earphone", "earphones",
        "earbud", "earbuds", "headphone", "headphones", "case", "device",
        "electronics", "gadget", "packaging", "商品图", "产品图", "商品",
        "产品", "耳机", "蓝牙耳机", "充电盒", "电子产品", "包装"
    ]):
        return {
            "type": "hard_product",
            "max_fill": 0.88,
            "max_merged_fill": 0.86,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 6,
            "require_overlap_for_attachments": False
        }

    if any(token in text for token in [
        "vases_flowers", "vase flowers", "vase and flowers", "flowers in vase",
        "flower arrangement", "bouquet", "plant arrangement", "potted plant",
        "pot plant", "arrangement", "花瓶花艺", "花艺", "插花", "花束", "盆栽", "植物组合"
    ]):
        return {
            "type": "decor_arrangement",
            "max_fill": 0.82,
            "max_merged_fill": 0.78,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 7,
            "require_overlap_for_attachments": False
        }

    if any(token in text for token in [
        "vase", "bowl", "sculpture", "ornament", "column", "stacked", "cylinder",
        "decor_vase", "decor_bowl", "decor_sculpture",
        "花瓶", "碗", "摆件", "雕塑", "柱状", "立柱", "叠柱", "装饰柱", "装饰碗"
    ]):
        return {
            "type": "decor_atomic",
            "max_fill": 0.74,
            "max_merged_fill": 0.64,
            "max_attachment_distance": 8,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 3,
            "require_overlap_for_attachments": True
        }

    if any(token in text for token in ["table", "desk", "coffee table", "茶几", "桌"]):
        return {
            "type": "table",
            "max_fill": 0.72,
            "max_merged_fill": 0.72,
            "max_attachment_distance": 8,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 4,
            "require_overlap_for_attachments": True
        }

    if any(token in text for token in ["chair", "sofa", "seat", "stool", "沙发", "椅", "凳"]):
        return {
            "type": "furniture",
            "max_fill": 0.78,
            "max_merged_fill": 0.68,
            "max_attachment_distance": 10,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 4,
            "require_overlap_for_attachments": True
        }

    return {
        "type": "default",
        "max_fill": MAX_TARGET_FILL_RATIO,
        "max_merged_fill": MAX_MERGED_TARGET_FILL_RATIO,
        "max_attachment_distance": MAX_ATTACHMENT_DISTANCE_PX,
        "allow_attachments": True,
        "prefer_rectangular": False,
        "max_masks": 4,
        "require_overlap_for_attachments": False
    }

def _is_completion_layer_meta(layer_meta):
    meta = layer_meta or {}
    return bool(
        meta.get("completionSegmentation") or
        str(meta.get("id") or "").startswith("completion-scene-sam-") or
        str(meta.get("layerId") or "").startswith("completion-scene-sam-")
    )


def infer_spatial_hard_entity_type(meta, text, base_type):
    """Resolve durable structural types without changing flat product routing."""
    if base_type in {"table", "furniture", "lighting"}:
        return base_type

    design_role = str((meta or {}).get("designRole", "")).lower()
    extraction_profile = str((meta or {}).get("extractionProfile", "")).lower()
    semantic_type = str((meta or {}).get("semanticType", "")).lower()
    is_scene_entity = (
        design_role in {"scene_object", "spatial_object", "interior_object"} or
        extraction_profile in {"multi_part_hard_object", "multi_part_hard_product"}
    )
    is_explicit_food = semantic_type in {"product_food", "product_drink"} or design_role == "product_image"
    if not is_scene_entity or is_explicit_food:
        return None

    if any(token in text for token in [
        "lamp", "chandelier", "pendant", "lighting", "light fixture",
        "吊灯", "灯具", "灯饰", "照明", "吸顶灯", "壁灯"
    ]):
        return "lighting"
    if any(token in text for token in [
        "table", "desk", "console", "sideboard", "cabinet", "counter",
        "茶几", "桌面", "台面", "边几", "矮几", "端景", "玄关", "桌", "柜", "几", "台"
    ]):
        return "table"
    if any(token in text for token in [
        "chair", "sofa", "seat", "stool", "furniture",
        "沙发", "椅", "凳", "座椅", "家具"
    ]):
        return "furniture"
    return None


def spatial_hard_entity_strategy(strategy_type):
    """Return the established strategy contract for a structural scene entity."""
    strategies = {
        "table": {
            "type": "table",
            "max_fill": 0.66,
            "max_merged_fill": 0.72,
            "max_attachment_distance": 7,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 3,
            "require_overlap_for_attachments": True
        },
        "furniture": {
            "type": "furniture",
            "max_fill": 0.70,
            "max_merged_fill": 0.58,
            "max_attachment_distance": 7,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 3,
            "require_overlap_for_attachments": True
        },
        "lighting": {
            "type": "lighting",
            "max_fill": 0.62,
            "max_merged_fill": 0.54,
            "max_attachment_distance": 6,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 3,
            "require_overlap_for_attachments": True
        }
    }
    return dict(strategies.get(strategy_type, {}))


def get_layer_strategy(layer_meta):
    """Resolve one coarse profile plus composable capabilities.

    ``type`` and the legacy thresholds remain available to existing mask
    helpers. New routing should use ``profile``, ``features``, and ``phase``;
    this keeps object labels from selecting an unrelated end-to-end pipeline.
    """
    meta = layer_meta or {}
    legacy = _get_legacy_layer_strategy(meta)
    base_type = legacy.get("type", "default")
    text = " ".join([
        str(meta.get("name", "")),
        str(meta.get("semanticType", "")),
        str(meta.get("designRole", "")),
        str(meta.get("category", "")),
        str(meta.get("runtimeType", "")),
        str(meta.get("extractionProfile", ""))
    ]).lower()

    spatial_types = {
        "table", "furniture", "lighting", "decor_arrangement", "decor_atomic"
    }
    spatial_tokens = [
        "interior", "room", "space", "furniture", "table", "desk", "console",
        "sideboard", "cabinet", "chair", "sofa", "stool", "lamp", "lighting",
        "curtain", "window", "窗", "窗纱", "纱帘",
        "室内", "空间", "家具", "桌", "台", "柜", "椅", "凳", "沙发", "灯具"
    ]
    profile = "spatial_design" if (
        base_type in spatial_types or any(token in text for token in spatial_tokens)
    ) else "flat_design"

    # Upstream metadata can describe a physical scene object as a generic hard
    # product or soft edge. Restore a durable structural type only for actual
    # scene entities; poster products stay on their existing flat route.
    spatial_hard_type = infer_spatial_hard_entity_type(meta, text, base_type)
    if profile == "spatial_design" and spatial_hard_type:
        base_type = spatial_hard_type
        legacy = spatial_hard_entity_strategy(base_type)

    phase = "completion" if _is_completion_layer_meta(meta) else "initial"
    features = set()
    if base_type == "soft_edge" or any(token in text for token in [
        "feather", "plume", "hair", "fur", "smoke", "cloud", "fog", "sheer",
        "curtain", "tulle", "transparent", "羽毛", "毛发", "烟", "云", "雾",
        "窗纱", "纱", "薄纱", "透明"
    ]):
        features.add("soft_edge")
    else:
        features.add("hard_edge")

    if base_type in {"food_product", "decor_arrangement"} or (
        str(meta.get("semanticType", "")).lower() == "decor_plant" and
        str(meta.get("designRole", "")).lower() == "scene_object"
    ) or any(token in text for token in [
        "compound", "arrangement", "plate", "dish", "meal", "food", "食物", "菜品",
        "餐盘", "组合", "花艺", "插花"
    ]):
        features.add("compound")
    if base_type in {"flat_shape"} or any(token in text for token in [
        "text", "label", "badge", "logo", "panel", "文字", "文本", "徽章", "标志"
    ]):
        features.add("text_overlap")
    if base_type in {"table", "furniture", "lighting"}:
        features.add("supports")
        # A material word such as "transparent" describes a glass tabletop,
        # not a feathered silhouette. Spatial hard objects must retain their
        # original prompt raster and safe-matte route.
        features.discard("soft_edge")
        features.add("hard_edge")
    if phase == "completion" or meta.get("completionOccluder"):
        features.add("occluded")

    # Flat completion uses one canonical selector. A physical spatial entity
    # keeps its first-pass strategy throughout completion so table/furniture
    # B/L recovery and safe matte are not replaced by poster-object rules.
    spatial_canonical_completion = bool(
        phase == "completion" and
        profile == "spatial_design" and
        base_type in {"table", "furniture", "lighting"}
    )
    execution = {
        "type": "completion_object",
        "max_fill": 0.90,
        "max_merged_fill": 0.90,
        "max_attachment_distance": 18,
        "allow_attachments": True,
        "prefer_rectangular": False,
        "max_masks": 8,
        "require_overlap_for_attachments": False
    } if phase == "completion" and not spatial_canonical_completion else legacy

    result = {
        **execution,
        "profile": profile,
        "features": sorted(features),
        "phase": phase,
        "baseStrategyType": base_type,
        "selectionType": (
            base_type if spatial_canonical_completion else
            "completion_object" if phase == "completion" else base_type
        ),
        "spatialCanonicalCompletion": spatial_canonical_completion
    }
    return result


def get_boundary_recovery_config(layer_meta=None, strategy=None):
    """Resolve boundary recovery as a capability, not an object category."""
    meta = layer_meta or {}
    strategy = strategy or get_layer_strategy(meta)
    features = set(strategy.get("features", []))
    phase = strategy.get("phase", "initial")
    edge_type = (
        "soft_edge" if "soft_edge" in features else
        "text" if "text_overlap" in features else
        "hard_edge"
    )
    # Text and soft silhouettes keep their established specialized alpha paths.
    # They still have an explicit edge_type here, so future recovery changes
    # can tune parameters without creating another routing branch.
    enabled = bool(
        phase == "initial" and
        edge_type == "hard_edge" and
        not bool(meta.get("completionOccluder")) and
        strategy.get("baseStrategyType") != "flat_shape"
    )
    return {
        "enabled": enabled,
        "edgeType": edge_type,
        "minTouchCount": 4,
        "lowFillThreshold": 0.42,
        "recoveryTriggerFill": 0.52,
        "disagreementTriggerFill": 0.52,
        "minAddedPixels": 32,
        "minAddedRatio": 0.025,
        "minPreserveCore": 0.90,
        "maxGrowth": 2.20,
        "maxPromptAreaRatio": 0.92,
        "minAttachedGrowth": 0.78,
        "maxContextConflictRatio": 0.16
    }


def get_semantic_ownership_prompt_config(layer_meta=None, strategy=None):
    """Resolve semantic ownership prompting as a shared capability."""
    meta = layer_meta or {}
    strategy = strategy or get_layer_strategy(meta)
    features = set(strategy.get("features", []))
    phase = strategy.get("phase", "initial")
    edge_type = (
        "soft_edge" if "soft_edge" in features else
        "text" if "text_overlap" in features else
        "hard_edge"
    )
    spatial_ownership = get_spatial_ownership_config(meta, strategy)
    instance_guard = meta.get("_spatialOwnershipGuard") or {}
    has_verified_atomic_sibling = bool(instance_guard.get("entries"))
    return {
        "enabled": bool(
            # Ownership is evidence-driven, not domain-driven. A flat poster
            # can contain overlapping raster objects just as an interior can.
            # The prompt compiler still refuses to emit points without safe
            # component/exclusion evidence, preserving the bbox-only baseline.
            (
                spatial_ownership["promptEnabled"] or
                (
                    spatial_ownership.get("atomicInstancePromptEligible") and
                    has_verified_atomic_sibling
                )
            ) and
            phase == "initial" and
            edge_type == "hard_edge" and
            strategy.get("baseStrategyType") != "flat_shape" and
            not bool(meta.get("completionOccluder"))
        ),
        "edgeType": edge_type,
        "includeBoundaryNegatives": False,
        "spatialOwnership": spatial_ownership
    }


def get_spatial_ownership_config(layer_meta=None, strategy=None):
    """Enable ownership guidance only for evidenced raster atomic assets.

    A sibling bbox is not a pixel mask.  This capability authorizes sparse SAM
    point prompts and locks output geometry only when Gemini supplied a
    high-confidence ownership contract. Legacy metadata without an explicit
    contract remains on the established September 16 bbox-only path.
    """
    meta = layer_meta or {}
    strategy = strategy or get_layer_strategy(meta)
    intent = meta.get("segmentationIntent")
    if not isinstance(intent, dict):
        return {
            "enabled": False,
            "promptEnabled": False,
            "strictOutput": False,
            "reason": "missing_intent"
        }
    try:
        confidence = float(intent.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    included = intent.get("includedComponents")
    excluded = intent.get("excludedAdjacentObjects")

    def box_area(box):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return 0.0
        try:
            ymin, xmin, ymax, xmax = [float(value) for value in box]
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, ymax - ymin) * max(0.0, xmax - xmin)

    def intersection_area(first, second):
        if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
            return 0.0
        try:
            ymin = max(float(first[0]), float(second[0]))
            xmin = max(float(first[1]), float(second[1]))
            ymax = min(float(first[2]), float(second[2]))
            xmax = min(float(first[3]), float(second[3]))
        except (TypeError, ValueError, IndexError):
            return 0.0
        return max(0.0, ymax - ymin) * max(0.0, xmax - xmin)

    # A component rectangle that is wholly covered by the target rectangle
    # cannot provide a safe negative ownership point.  Treating that metadata
    # as strict evidence is what incorrectly routed flat raster layers into
    # tight-subject path.  Require either multiple included components or an
    # exclusion with a verifiable region outside the included components.
    safe_exclusion = False
    for entry in excluded if isinstance(excluded, list) else []:
        entry_box = entry.get("bbox") if isinstance(entry, dict) else None
        entry_area = box_area(entry_box)
        if entry_area <= 0:
            continue
        covered_area = sum(
            intersection_area(entry_box, component.get("bbox"))
            for component in (included if isinstance(included, list) else [])
            if isinstance(component, dict)
        )
        if entry_area - min(entry_area, covered_area) >= 16:
            safe_exclusion = True
            break
    ownership_evidence = bool(
        isinstance(included, list) and len(included) >= 2
    ) or safe_exclusion
    enabled = bool(
        strategy.get("phase") == "initial" and
        "hard_edge" in set(strategy.get("features", [])) and
        intent.get("targetKind") in {"atomic_object", "composite_assembly"} and
        isinstance(included, list) and len(included) > 0 and
        confidence >= 0.85 and
        ownership_evidence
    )
    # An atomic asset can describe its own physical parts (for example a
    # tabletop and its legs). That alone is not evidence to change SAM's
    # bbox-only proposal. It additionally needs a runtime-verified sibling.
    atomic_instance_prompt_eligible = bool(
        enabled and
        intent.get("targetKind") == "atomic_object" and
        intent.get("bboxRole") == "tight_subject" and
        safe_exclusion
    )
    # Composite assemblies can use their declared component anchors directly.
    # Atomic prompts remain disabled until the API proves the exclusion matches
    # an actual nearby context sibling.
    prompt_enabled = bool(
        enabled and
        intent.get("targetKind") == "composite_assembly" and
        isinstance(included, list) and
        len(included) >= 2
    )
    return {
        "enabled": enabled,
        "promptEnabled": prompt_enabled,
        "atomicInstancePromptEligible": atomic_instance_prompt_eligible,
        # A tight subject is a valid output boundary only when the semantic
        # contract itself is present.  The core checks candidate containment
        # before enforcing it, so an incomplete box cannot cut a chair/sofa.
        "strictOutput": bool(enabled and intent.get("bboxRole") == "tight_subject"),
        "reason": "accepted" if enabled else (
            "no_safe_ownership_evidence" if confidence >= 0.85 else
            "not_confident_atomic_intent"
        ),
        "confidence": round(confidence, 3),
        "bboxRole": intent.get("bboxRole")
    }


def get_positive_probe_config(layer_meta=None, strategy=None):
    """Resolve positive-point recovery from capabilities, never categories."""
    meta = layer_meta or {}
    strategy = strategy or get_layer_strategy(meta)
    features = set(strategy.get("features", []))
    phase = strategy.get("phase", "initial")
    edge_type = (
        "soft_edge" if "soft_edge" in features else
        "text" if "text_overlap" in features else
        "hard_edge"
    )
    return {
        "enabled": bool(
            phase == "initial" and
            edge_type == "hard_edge" and
            strategy.get("baseStrategyType") != "flat_shape" and
            not bool(meta.get("completionOccluder"))
        ),
        "edgeType": edge_type,
        "minPreserveCore": 0.98,
        "minFillGain": 0.025,
        "minGrowthGain": 0.025,
        "maxGrowth": 1.45,
        "minConnectedGrowth": 0.35,
        "maxOutsideAddedRatio": 0.55,
        "maxContextConflictRatio": 0.12
    }


# The legacy type names below are implementation details kept for compatibility
# with the existing matte helpers. Routing and quality decisions must use this
# policy instead of growing another object-name branch.
MASK_POLICY_VERSION = "profiles-v2-phase3.2"


def resolve_mask_policy(layer_meta=None, quality_profile="publish"):
    """Resolve the small set of capabilities shared by all segmentation paths."""
    strategy = get_layer_strategy(layer_meta or {})
    features = set(strategy.get("features", []))
    phase = strategy.get("phase", "initial")
    profile = strategy.get("profile", "flat_design")
    base_type = strategy.get("baseStrategyType", strategy.get("type", "default"))
    completion = phase == "completion"
    soft_edge = "soft_edge" in features
    completion_occluder = bool(
        (layer_meta or {}).get("completionOccluder") and
        phase == "initial" and
        "soft_edge" not in features
    )
    compound = "compound" in features
    spatial = profile == "spatial_design"
    spatial_canonical_completion = bool(strategy.get("spatialCanonicalCompletion"))
    boundary_recovery = get_boundary_recovery_config(
        layer_meta or {},
        strategy=strategy
    )
    semantic_ownership_prompt = get_semantic_ownership_prompt_config(
        layer_meta or {},
        strategy=strategy
    )
    # A tight atomic spatial asset with an explicit sibling exclusion must not
    # trigger the optional bbox-expanding L recovery route.  Its normal B mask
    # and positive probe remain available inside the original semantic bbox.
    spatial_ownership = get_spatial_ownership_config(layer_meta or {}, strategy)
    if spatial_ownership.get("enabled"):
        boundary_recovery = {**boundary_recovery, "enabled": False}
    positive_probe = get_positive_probe_config(
        layer_meta or {},
        strategy=strategy
    )
    compound_component_discovery = {
        # Composition is a capability signal. It does not change the
        # established person/object recovery thresholds or routes.
        "enabled": bool(
            compound and
            phase == "initial" and
            boundary_recovery.get("edgeType") == "hard_edge" and
            not completion_occluder
        ),
        "promptExpandRatio": 0.36,
        "minCorePreserve": 0.97,
        "minComponentAreaRatio": 0.0025,
        "maxGrowthRatio": 0.75,
        "maxContextConflictRatio": 0.22,
        "maxContextComponentAreaRatio": 0.02,
        "maxStrictOwnershipConflictRatio": 0.18,
        "maxDetachedDistance": 18,
        "minComponentFill": 0.42,
        "maxEnvelopeFill": 0.40,
        "maxPromptEdgeFill": 0.55,
        "minContainedEdgeEnvelopeTargetRatio": 0.95,
        "targetEdgeMarginRatio": 0.05,
        "minEdgeEnvelopeHeightRatio": 0.35,
        "minEdgeEnvelopeWidthRatio": 0.35
    }
    interior_hole_recovery = {
        # This is an evidence-gated capability, not a category route. It is
        # evaluated for hard-edge SAM masks and remains a no-op for ordinary
        # masks without substantial enclosed holes.
        "enabled": bool(
            phase == "initial" and
            boundary_recovery.get("edgeType") == "hard_edge" and
            not completion_occluder
        ),
        "minHolePixels": 64,
        "minHoleRatio": 0.008,
        # Individual holes can be fragmented by image texture or a coarse
        # SAM contour. Filter only tiny speckles per component, then apply the
        # substantial-hole threshold to their aggregate area.
        "minHoleComponentPixels": 8,
        "minRingSupport": 0.72,
        "maxHoleCount": 24,
        "maxProbePoints": 4,
        "minHoleFillRatio": 0.25,
        # SAM's positive-point mask may include a broad surrounding object
        # envelope while still proving that the audited hole belongs to the
        # subject. The final merge clips that verified growth to the audited
        # hole union, so a high-confidence hole fill can be accepted without
        # importing the envelope.
        "minClippedHoleFillRatio": 0.90,
        "minClippedCandidateCoreOverlap": 0.90,
        "minPreserveCore": 0.98,
        "maxAddedRatio": 0.16,
        "maxOutsideHoleRatio": 0.18,
        "maxStrictOwnershipRatio": 0.18
    }
    # SAM-B is the primary instance/coverage proposal.  SAM-L can provide a
    # more precise hard boundary, but must not replace that proposal wholesale:
    # its edits are later limited to a narrow B-derived boundary band and are
    # accepted only when they preserve B's stable core and topology.  This is
    # deliberately based on visual capability and execution phase, rather
    # than labels such as chair, person, food, or product.
    cross_model_boundary_refine = {
        # Retained as a compatibility capability for old diagnostics, but the
        # former full-image B-accepted -> L arbitration is intentionally off.
        # Accepted furniture uses the bounded local-L capability below.
        "enabled": False,
        "bandRatio": 0.018,
        "minStableCorePreserve": 0.985,
        "minBaselinePreserve": 0.985,
        "minLCoverageOfBaseline": 0.94,
        "maxChangedRatio": 0.075,
        "maxRemovedRatio": 0.015,
        "maxAddedRatio": 0.06
    }
    local_l_refine = {
        # Keep the ROI implementation for later experiments, but disable it
        # in the production route. L is reserved for B-failure recovery.
        "enabled": False,
        "roiExpandRatio": 0.18,
        "bandRatio": 0.024,
        "minCorePreserve": 0.985,
        "minBaselinePreserve": 0.985,
        "minCandidateOverlap": 0.94,
        "maxChangedRatio": 0.10,
        "maxRemovedRatio": 0.015,
        "maxAddedRatio": 0.10
    }
    instance_guard = (layer_meta or {}).get("_spatialOwnershipGuard") or {}
    ownership_boundary_repair = {
        # This is image-only alpha geometry repair, not a second instance
        # decision. It is available only after a verified atomic sibling guard
        # has separated the subject from an adjacent instance.
        "enabled": bool(
            phase == "initial" and
            boundary_recovery.get("edgeType") == "hard_edge" and
            spatial_ownership.get("atomicInstancePromptEligible") and
            isinstance(instance_guard.get("entries"), list) and
            len(instance_guard.get("entries")) > 0 and
            not completion_occluder
        ),
        "bandRatio": .024,
        "maxBandPixels": 8,
        "minAgreement": .98,
        "minCoreRatio": .40,
        "backgroundOdds": 4,
        # The image review may be broader internally, but this ownership
        # adapter outputs additions only and caps them to 2.5% of the mask.
        "maxAddedRatio": .025,
        "maxRemovedRatio": .10,
        "minComponentPreserve": .90,
        "maxReviewPixels": 1500000
    }
    # A semantic composite is intentionally one workbench layer.  Its SAM
    # output must therefore be allowed to contain multiple disconnected
    # instances; do not confuse this with splitting the layer into children.
    composite_instance_union = bool(
        (layer_meta or {}).get("compositeRole") == "composite_group" or
        (isinstance((layer_meta or {}).get("childLayerIds"), list) and
         len((layer_meta or {}).get("childLayerIds")) > 0)
    )

    if spatial_canonical_completion:
        selector = "spatial_canonical_completion"
        selection_type = base_type
    elif completion:
        selector = "generic_completion"
        selection_type = "completion_object"
    elif compound and base_type == "food_product":
        selector = "compound_food"
        selection_type = "food_product"
    else:
        selector = "profile_object"
        selection_type = base_type

    # Keep the established high-resolution spatial/soft-edge paths. Flat
    # products use 1024 unless a later quality decision escalates them to L.
    # Flat completion uses the stable completion_object route. Spatial
    # canonical completion retains its original table/furniture sizing policy.
    sam_imgsz = 1024 if completion and not spatial_canonical_completion else (
        SOFT_EDGE_SAM_IMGSZ if soft_edge else
        HARD_EDGE_SAM_IMGSZ if base_type in HARD_EDGE_STRATEGIES else
        1024
    )
    # A spatial canonical completion carries its original table/furniture
    # alpha policy through the second SAM pass.
    matte_type = (
        base_type if spatial_canonical_completion else
        base_type if completion and base_type in {"table", "furniture"} else
        selection_type if completion else (
        "soft_edge" if soft_edge else
        base_type if base_type in {"table", "furniture"} else
        selection_type
        )
    )
    return {
        "version": MASK_POLICY_VERSION,
        "profile": profile,
        "phase": phase,
        "features": sorted(features),
        "baseStrategyType": base_type,
        "selectionType": selection_type,
        "selector": selector,
        "matteType": matte_type,
        "softEdge": soft_edge,
        "compound": compound,
        "spatial": spatial,
        "completion": completion,
        "spatialCanonicalCompletion": spatial_canonical_completion,
        "boundaryRecovery": boundary_recovery,
        "semanticOwnershipPrompt": semantic_ownership_prompt,
        "spatialOwnership": spatial_ownership,
        "positiveProbe": positive_probe,
        "compoundComponentDiscovery": compound_component_discovery,
        "interiorHoleRecovery": interior_hole_recovery,
        "crossModelBoundaryRefine": cross_model_boundary_refine,
        "localLRefine": local_l_refine,
        "ownershipBoundaryRepair": ownership_boundary_repair,
        "boundaryMatte": {
            # Existing authoritative spatial mattes, soft alpha, composite
            # masks and completion keep their own pixel ownership contracts.
            "enabled": bool(
                positive_probe["enabled"] and not spatial and
                not composite_instance_union and
                str(os.environ.get("SAM_BOUNDARY_MATTE", "1")).lower() not in {"0", "false", "no"}
            ),
            "bandRatio": .03,
            "minAgreement": .98,
            # Compound masks may contain plates and attached components with
            # low-contrast rims. A matte that keeps only 68% of the accepted
            # SAM core is destructive even when its two radius outputs agree.
            # Keep the broader threshold for all existing non-compound paths.
            "minCoreRatio": .90 if compound else .40,
            "backgroundOdds": 4,
            "maxAddedRatio": .10,
            "maxRemovedRatio": .10,
            "minComponentPreserve": .90,
            "maxReviewPixels": 1500000
        },
        "completionOccluder": completion_occluder,
        "compositeInstanceUnion": composite_instance_union,
        "samImgSize": sam_imgsz,
        "allowLocalRefine": (
            not spatial_canonical_completion and
            (completion or not (soft_edge or compound or spatial))
        ),
        "allowLocalUpscale": completion or not (soft_edge or compound),
        # Initial compound objects keep their established B-first route unless
        # their alpha will define a destructive scene-inpaint mask. Those
        # foreground occluders receive an independent SAM-L review regardless
        # of semantic name or product category.
        "allowModelEscalation": (
            True if completion or completion_occluder
            else (
                not soft_edge and
                (
                    not compound or
                    cross_model_boundary_refine.get("enabled", False)
                )
            )
        ),
        "qualityProfile": normalize_sam_quality_profile(quality_profile)
    }


def policy_for_strategy_type(strategy_type=None, layer_meta=None, quality_profile="publish"):
    """Return a policy even for old callers that only pass a strategy string."""
    if layer_meta is not None:
        return resolve_mask_policy(layer_meta, quality_profile)
    strategy_type = str(strategy_type or "default")
    return resolve_mask_policy({
        "semanticType": strategy_type,
        "extractionProfile": strategy_type,
        "completionSegmentation": strategy_type == "completion_object"
    }, quality_profile)
