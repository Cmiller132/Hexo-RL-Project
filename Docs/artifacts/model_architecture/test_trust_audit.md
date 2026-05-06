# Test Trust Audit

Classification:

- `golden`: trusted behavior to preserve after harness dependencies are fixed.
- `rewrite`: useful intent but current assertion or dependency must change.
- `delete`: legacy behavior only after a replacement test exists.

## Suite-Level Audit

| Test path | Behavior claimed | Actual boundary covered | Classification | Known blind spots | Replacement or extension |
|---|---|---|---|---|---|
| `Python/tests/test_config_and_guardrails.py` | config validation, head/loss guardrails, architecture smoke, pair strategy gating | pydantic schema, `HexNet` construction, some loss errors | golden with Stage 2 rewrites | config currently owns architecture authority and mutates graph defaults | registry/spec override tests; self-play required output tests |
| `Python/tests/test_training_data_pipeline.py` | replay, target building, candidate/pair rows, loss functions, epoch glue | buffer/ring/sampler/targets/losses and some pipeline flows | mixed: golden plus rewrite | sparse/graph sampler tests fail without `_engine`; one loss test preserves silent skips | row/target contract tests; rewrite silent skip test |
| `Python/tests/test_global_graph_contract.py` | global graph row semantics, pair rows, graph model forward/trainer | graph batch, `GlobalHexGraphNet`, `compute_losses` | golden intent, rewrite harness | 30/35 fail when `_engine` missing; tests use legacy model authority | contract-local fixtures or built engine; move to `hexorl.models` authority |
| `Python/tests/test_inference_server.py` | dense/sparse/graph inference queue behavior | server/client shared-memory and model decoding | mixed | graph test fails without `_engine`; no protocol row hash checks | protocol adapter tests in Stage 4 |
| `Python/tests/test_tactical_oracle.py` | tactical oracle behavior and production fallback policy | oracle wrapper and candidate builder | golden | production requires `_engine`, skipped when unavailable | retain; add protocol source labels for tactical targets |
| `Python/tests/test_engine_smoke.py` | Rust engine API and MCTS guards | `_engine` extension | golden but unavailable | all skipped when `_engine` missing | CI must build Rust extension for phase closure |
| `Python/tests/test_engine_invariants.py` | Rust/Python legal, winner, D6, oracle invariants | `_engine` extension and Python references | golden but unavailable | all skipped when `_engine` missing | CI must build Rust extension for phase closure |
| `Python/tests/test_production_smoke.py` | tiny production pipeline smoke | epoch, self-play, training, checkpoint | rewrite for model architecture closure | broad smoke; expensive; depends on runtime availability | keep as tiered smoke, not contract proof |
| `Python/tests/test_dashboard_foundation.py` | dashboard storage/render/play/model diagnostics | dashboard endpoints and local model helpers | mostly golden outside model refactor | dashboard pair inference may use raw head semantics | add adapter-facing dashboard tests |
| `Python/tests/test_dashboard_replay_debug.py` | replay debug endpoint payload | dashboard replay API | golden outside model refactor | not a model contract proof | keep for dashboard |
| `Python/tests/test_rgsc_restart_service.py` | RGSC restart service and PRB behavior | self-play restart services | golden outside model refactor | does not prove model architecture contracts | keep for RGSC |
| `Python/tests/test_phase3_scorecard.py` | autotune scorecard and tactical suite | eval/tuning scoring | golden outside model refactor | no model boundary proof | keep |
| `Python/tests/test_phase3_autotune.py` | scheduler/search config/tuning behavior | tuning manifests and runtime sweeps | mixed | contains global graph config assumptions that will move to specs | rewrite architecture family assertions after Stage 2 |
| `Python/tests/test_checkpoint_league.py` | checkpoint league persistence/eval | eval league | golden outside model refactor | no model contract proof | keep |

## Per-Test Inventory For Model Refactor Boundary

This section is the Stage 2 working list. Classifications apply to individual
tests, not just files.

### `Python/tests/test_config_and_guardrails.py`

| Classification | Tests | Stage 2/3 treatment |
|---|---|---|
| `golden` | `test_config_rejects_lookahead_head_without_matching_horizon`; `test_config_requires_active_loss_for_matching_lookahead_head`; `test_config_forbids_unknown_fields`; `test_regret_fraction_requires_weighted_regret_heads_or_replay_only`; `test_config_rejects_mismatched_lookahead_horizon_and_lambda_counts`; `test_compute_losses_raises_when_no_loss_can_be_computed`; `test_ring_buffer_rejects_invalid_dimensions`; `test_model_ema_decay_keeps_most_shadow_weight`; `test_selfplay_worker_game_ids_are_unique_across_workers`; `test_autotune_train_batch_avoids_memory_cliff_for_production_model`; `test_autotune_compile_model_for_long_cuda_training`; `test_restnet_config_validation_and_forward_shapes`; `test_graph_config_validation_and_action_keyed_forward_shapes`; `test_hex_conv_invalid_axial_corners_stay_zero_after_optimizer_step`; `test_hex_conv_masks_are_reapplied_after_loading_state_dict`; `test_trunks_use_hex_conv_for_architecture_names`; `test_pair_policy_head_forward_and_default_weight`; `test_restnet_config_rejects_invalid_attention_position`; `test_sparse_policy_config_requires_active_loss_for_sparse_policy_head`; `test_sparse_policy_head_enables_sparse_data_contract`; `test_cnn_config_does_not_require_attention_head_divisibility`; `test_config_rejects_reserved_or_invalid_attention_options`; `test_sparse_policy_effective_candidate_width_capped_by_shm`; `test_sparse_prior_stage_requires_sparse_policy_contract`; `test_rgsc_selfplay_config_bounds` | Preserve behavior; move architecture assertions from config/model classes to registry/spec tests where applicable. |
| `rewrite` | `test_graph_architecture_alias_is_not_runtime_supported`; `test_global_xattn_pair_strategy_defaults_to_none`; `test_global_xattn_pair_heads_do_not_enable_pair_scoring_without_strategy`; `test_pair_scoring_requires_explicit_diagnostic_strategy_and_cap` | Keep the intent, but assert registry alias policy and executable pair-strategy contracts instead of legacy config mutation. |

### `Python/tests/test_training_data_pipeline.py`

| Classification | Tests | Stage 2/3 treatment |
|---|---|---|
| `golden` | `test_python_decoder_returns_final_position_for_history`; `test_policy_symmetry_transform_tracks_dense_target`; `test_tensor_and_policy_symmetry_match_for_all_transforms`; `test_axis_label_symmetry_transform_remains_valid`; `test_each_symmetry_permutates_axes_one_to_one`; `test_axis_delta_maps_symmetry_transforms_space_and_axis_planes`; `test_hexo_turn_boundaries_follow_player_runs`; `test_random_histories_have_stable_hexo_turn_starts`; `test_value_from_source_perspective_flips_opponent_values`; `test_lookahead_flips_future_player_perspective`; `test_lookahead_keeps_same_player_perspective`; `test_ema_lookahead_uses_source_perspective_for_every_future_term`; `test_mid_turn_lookahead_targets_next_turn_start`; `test_opponent_policy_uses_next_full_search_opponent_turn_start`; `test_opponent_policy_ignores_low_pcr_opponent_turn`; `test_opponent_policy_end_of_game_without_future_turn_zeroes_weight`; `test_regret_uses_selected_action_value_and_raw_scale`; `test_regret_weight_zero_when_selected_action_value_missing`; `test_regret_suffix_average_matches_paper_equation_2`; `test_compute_regret_requires_selected_action_value_by_default`; `test_truncated_games_zero_regret_weight`; `test_truncated_games_keep_policy_targets_but_zero_value_weight`; `test_draw_games_keep_policy_targets_but_zero_value_weight`; `test_process_game_record_populates_auxiliary_targets`; `test_ring_buffer_preserves_auxiliary_targets`; `test_ring_buffer_preserves_missing_selected_action_and_regret_weight`; `test_policy_target_v2_preserves_outside_window_mass`; `test_compact_record_v2_roundtrip_preserves_global_targets`; `test_compact_record_preserves_missing_selected_action_as_invalid_regret_target`; `test_ring_buffer_preserves_policy_target_v2`; `test_dense_projection_uses_all_v2_visits_before_topk`; `test_ring_buffer_truncates_primary_v2_targets_to_compact_width`; `test_replay_dataset_emits_regret_weight_for_loss_masking`; `test_regret_biased_sampling_ignores_zero_weight_regret_rows`; `test_candidate_feature_names_match_tensor_width`; `test_checkpoint_reports_candidate_feature_version`; `test_sparse_d6_batch_trains_for_all_model_architectures`; `test_pair_policy_targets_use_full_policy_v2_by_default`; `test_pair_policy_d6_bijection_preserves_pair_identity`; `test_pair_policy_rejects_duplicate_and_illegal_pairs`; `test_pair_candidate_builder_ignores_padded_candidate_rows`; `test_candidate_builder_accepts_list_legal_moves`; `test_candidate_builder_keeps_critical_actions_past_budget`; `test_critical_actions_are_inserted_before_heuristic_candidates`; `test_candidate_recall_reports_protected_and_discovery_modes`; `test_discovery_recall_does_not_include_target_only_actions`; `test_candidate_features_do_not_include_policy_target_labels`; `test_first_placement_pair_target_requires_recorded_joint_table`; `test_sparse_policy_loss_masks_invalid_candidates`; `test_sparse_policy_loss_accepts_half_logits_float_targets`; `test_policy_and_value_losses_accept_half_logits`; `test_policy_target_top64_is_preserved_when_configured`; `test_compact_replay_estimate_scales_to_200k_samples`; `test_run_epoch_appends_selfplay_to_existing_replay`; `test_selfplay_epoch_completion_requires_games_and_states`; `test_orchestrator_stop_is_idempotent_with_missing_worker_slot`; `test_orchestrator_masks_truncated_game_value_targets`; `test_policy_loss_can_be_masked_to_full_search_samples`; `test_opp_policy_loss_uses_opponent_policy_weight`; `test_opp_policy_loss_skips_empty_targets`; `test_value_loss_can_be_masked_for_truncated_games`; `test_value_loss_ignores_non_finite_targets_with_zero_weight`; `test_regret_losses_can_be_masked_by_regret_weight`; `test_axis_delta_norm_head_shape`; `test_replay_dataset_can_emit_axis_delta_norm_target`; `test_replay_dataset_marks_low_sim_policy_weight_zero`; `test_bootstrap_games_are_diverse_and_legal` | Preserve as target/replay/training semantics. Some should move to contract-local fixtures, but their assertions remain trusted. |
| `rewrite` | `test_sparse_sampler_outputs_candidate_targets`; `test_replay_dataset_can_emit_pair_policy_target`; `test_replay_dataset_second_placement_pair_target_keeps_known_first_row`; `test_graph_replay_can_emit_full_first_placement_pair_rows`; `test_critical_overflow_zeroes_sparse_and_pair_signal`; `test_sparse_sampler_preserves_all_targets_when_capacity_is_sufficient`; `test_graph_pair_training_rejects_incomplete_first_placement_pair_target` | Intent is golden, but the current harness fails without `_engine`; Stage 2/3 needs built-engine CI or explicit graph row fixtures that do not weaken production requirements. |
| `rewrite` | `test_compute_losses_skips_missing_targets_and_handles_batch_one` | Replace in Stage 3 with hard-error tests for missing trainable target/mask/weight/phase, plus a separate optional diagnostic omission test. |

### `Python/tests/test_global_graph_contract.py`

| Classification | Tests | Stage 2/3 treatment |
|---|---|---|
| `golden` | `test_global_graph_policy_logits_align_to_rust_legal_order`; `test_global_graph_policy_alignment_rejects_true_set_mismatch`; `test_global_graph_ipc_capacity_allows_full_legal_scout_requests`; `test_global_graph_rejects_sub_rust_legal_radius`; `test_global_graph_alternatives_have_distinct_model_families` | Preserve exact semantic intent; Stage 2 must source architecture ids/families from registry, not `GlobalHexGraphNet.ARCHITECTURES`. |
| `rewrite` | `test_global_graph_builder_preserves_all_legal_rows`; `test_global_graph_opponent_policy_uses_independent_legal_rows`; `test_global_graph_targets_must_match_their_own_legal_tables`; `test_global_graph_builder_includes_required_token_families_and_relations`; `test_global_graph_features_expose_rich_token_family_fields`; `test_global_graph_capacity_report_fails_without_dropping_rows`; `test_global_graph_relation_bias_contract_includes_cover_pair_and_component_edges`; `test_global_graph_pair_rows_mask_opening_and_exist_on_two_placement_turns`; `test_global_graph_second_placement_pair_targets_are_ordered_and_conditional`; `test_global_graph_pair_targets_reject_duplicate_and_illegal_pairs`; `test_global_graph_pair_rows_fail_instead_of_silent_truncation`; `test_global_graph_pair_chunks_remove_ipc_pair_cap_as_semantic_limit`; `test_global_graph_reference_pair_rows_cover_full_first_placement_table`; `test_global_graph_model_forward_with_padded_batch`; `test_global_graph_trainer_runs_graph_native_step_without_dense_policy`; `test_global_graph_pair_second_loss_is_known_first_only`; `test_global_graph_pair_logits_mask_invalid_rows_even_without_pair_token_indices`; `test_global_graph_full_requires_relation_bias_contract`; `test_global_graph_relation_tensor_shapes_are_validated`; `test_global_graph_alternatives_share_targets_and_masks`; `test_global_graph_output_heads_gate_optional_work`; `test_global_graph_pair_heads_are_distinct_first_second_and_joint_contracts`; `test_global_graph_pair_joint_is_symmetric_for_unordered_rows`; `test_d6_graph_token_relation_pair_equivariance` | Golden intent, but rewrite harness/imports around `_engine`, `build_graph_batch_from_history`, legacy model classes, and loss-plan authority. |

### `Python/tests/test_inference_server.py`

| Classification | Tests | Stage 4 treatment |
|---|---|---|
| `golden` | `test_server_starts_and_stops`; `test_single_client_round_trip`; `test_server_forward_returns_sparse_pair_logits`; `test_adaptive_batching_two_clients`; `test_non_finite_outputs_are_sanitized_before_mcts`; `test_mcts_round_trip` | Preserve server lifecycle, queueing, dense/sparse response, sanitization, and MCTS protocol behavior. |
| `rewrite` | `test_server_forward_graph_returns_keyed_logits` | Golden intent, but rewrite after protocol carries row hashes, output masks, graph row identity, value decoder, and pair phase. Current harness is `_engine`-gated. |

### Self-Play, Smoke, And Runtime-Adjacent Tests

| Classification | Tests | Treatment |
|---|---|---|
| `golden` | `Python/tests/test_rgsc_restart_service.py::test_rgsc_restart_restores_current_player_and_turn_phase`; `test_rgsc_restart_rejects_illegal_or_stale_history`; `test_rgsc_restart_samples_from_prb_when_beta_one`; `test_prb_ema_update_after_restart_game`; `test_rgsc_tree_node_states_can_enter_prb`; `test_rgsc_tree_node_source_is_persisted_honestly`; `test_prb_sampling_uses_rank_score_not_ema_regret`; `test_prb_eviction_prefers_lower_ema_regret_then_oldest_sampled` | Preserve as self-play restart/PRB behavior; not a model architecture contract proof. |
| `rewrite` | `Python/tests/test_production_smoke.py::test_tiny_production_pipeline_records_games_metrics_and_checkpoint` | Keep as tiered smoke, but do not use as Stage 2 proof. Stage 2 needs focused registry/runtime-consumer tests first. |
| `golden` | `Python/tests/test_tactical_oracle.py` tests | Preserve tactical oracle behavior; skipped production cases require built `_engine` in CI before phase closure. |
| `golden unavailable` | `Python/tests/test_engine_smoke.py` tests; `Python/tests/test_engine_invariants.py` tests | Preserve as Rust boundary proof; current local environment skips because `_engine` is unavailable. |
| `mixed rewrite` | `Python/tests/test_phase3_autotune.py` global-graph architecture assertions | Rewrite any id-list/family assertions to registry capability filters. The current `global_graph768_champion` exclusion assertion is not a model contract. |
| `golden outside boundary` | dashboard, scorecard, and checkpoint-league tests listed in the suite audit | Keep for their domains; add adapter-facing model contract tests rather than stretching these suites. |

## Specific Rewrite Items

| Test | Current issue | New test |
|---|---|---|
| `test_compute_losses_skips_missing_targets_and_handles_batch_one` | Claims silent skip behavior that Stage 3 deletes for trainable heads | `test_loss_plan_errors_on_missing_trainable_target`; separate optional diagnostic test |
| graph tests using `build_graph_batch_from_history` | Fail when `_engine` is unavailable because tactical oracle production path is hard-required | Use built `_engine` in CI, or explicit contract fixture with oracle labels supplied without changing runtime production requirement |
| `test_graph_architecture_alias_is_not_runtime_supported` | Proves `graph` is a deleted alias, not runtime behavior | Config and registry raise a hard unsupported-id error |
| global graph config tests in tuning suite | Assert architecture strings/config mutation | Assert registry spec resolution and feature flags |
| opponent policy fallback tests if any are added | Would preserve alias target fallback | Assert independent `opp_policy_target` row contract |

## Golden Test List For Stage Closure

Stage 2 cannot close unless:

- architecture registry tests pass for every kept architecture id;
- alias/deletion behavior is tested for `graph`;
- config override and self-play required-output tests pass;
- import audit proves `hexorl/model` is not runtime authority.

Stage 3 cannot close unless:

- row, target, loss-plan, missing-data, zero-mass, duplicate-row, pair-phase,
  and lookahead no-fallback negative tests pass;
- dense, sparse, graph, and pair training integration tests pass.

Stage 4 cannot close unless:

- inference protocol row-hash/value-decoder/pair-phase tests pass;
- self-play providers and pair strategies run without raw head-name checks;
- shared-memory transport schema tests prove semantics are carried by protocol.
