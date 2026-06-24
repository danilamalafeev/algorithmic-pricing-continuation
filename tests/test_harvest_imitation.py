from __future__ import annotations

import pytest
import torch

from experiments.harvest_imitation import (
    HarvestImitationDataset,
    HarvestPhase,
    behavior_cloning_cross_entropy,
    classification_metrics,
    evaluate_behavior_cloning,
    inverse_frequency_class_weights,
    phase_for_teacher_option,
    temporal_train_validation_split,
    train_behavior_cloning,
)


OPTION_NAMES = (
    "HOLD_HIGH",
    "HARVEST_UNDERCUT",
    "MATCH_HIGH",
    "PUNISH_NASH",
    "PUNISH_LOW",
    "RESET_HIGH",
)


def _dataset() -> HarvestImitationDataset:
    features = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    option_labels = torch.tensor([0, 1, 3, 2, 1, 5])
    return HarvestImitationDataset.from_option_labels(
        features,
        option_labels,
        OPTION_NAMES,
        time_index=torch.tensor([30, 10, 50, 20, 60, 40]),
    )


@pytest.mark.parametrize(
    ("option", "phase"),
    [
        ("HOLD_HIGH", HarvestPhase.TEACH),
        ("MATCH_HIGH", HarvestPhase.TEACH),
        ("HARVEST_UNDERCUT", HarvestPhase.HARVEST),
        ("HARVEST_UNDERCUT_1", HarvestPhase.HARVEST),
        ("HARVEST_UNDERCUT_2", HarvestPhase.HARVEST),
        ("PUNISH_NASH", HarvestPhase.RECOVER),
        ("PUNISH_LOW", HarvestPhase.RECOVER),
        ("RESET_HIGH", HarvestPhase.RECOVER),
    ],
)
def test_scripted_and_learned_options_map_to_reusable_phases(
    option: str, phase: HarvestPhase
) -> None:
    assert phase_for_teacher_option(option) is phase


def test_unknown_teacher_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown teacher option"):
        phase_for_teacher_option("UNKNOWN")


def test_dataset_derives_phase_labels_and_validates_alignment() -> None:
    dataset = _dataset()
    torch.testing.assert_close(
        dataset.phase_labels,
        torch.tensor(
            [
                HarvestPhase.TEACH,
                HarvestPhase.HARVEST,
                HarvestPhase.RECOVER,
                HarvestPhase.TEACH,
                HarvestPhase.HARVEST,
                HarvestPhase.RECOVER,
            ]
        ),
    )
    with pytest.raises(ValueError, match="do not match"):
        HarvestImitationDataset(
            features=dataset.features,
            phase_labels=torch.zeros(6, dtype=torch.int64),
            option_labels=dataset.option_labels,
            option_names=dataset.option_names,
        )


def test_temporal_split_is_deterministic_and_uses_latest_rows_for_validation() -> None:
    dataset = _dataset()
    train_a, validation_a = temporal_train_validation_split(
        dataset, train_fraction=2 / 3
    )
    train_b, validation_b = temporal_train_validation_split(
        dataset, train_fraction=2 / 3
    )

    torch.testing.assert_close(train_a.time_index, torch.tensor([10, 20, 30, 40]))
    torch.testing.assert_close(validation_a.time_index, torch.tensor([50, 60]))
    torch.testing.assert_close(train_a.features, train_b.features)
    torch.testing.assert_close(validation_a.option_labels, validation_b.option_labels)


def test_temporal_split_without_time_index_preserves_row_order() -> None:
    source = _dataset()
    dataset = HarvestImitationDataset.from_option_labels(
        source.features, source.option_labels, source.option_names
    )
    train, validation = temporal_train_validation_split(dataset, train_fraction=0.5)
    torch.testing.assert_close(train.features, source.features[:3])
    torch.testing.assert_close(validation.features, source.features[3:])


def test_inverse_frequency_weights_balance_present_classes() -> None:
    labels = torch.tensor([0, 0, 0, 1, 2, 2])
    weights = inverse_frequency_class_weights(labels, num_classes=4)
    torch.testing.assert_close(
        weights, torch.tensor([2 / 3, 2.0, 1.0, 0.0])
    )
    weighted_counts = torch.bincount(labels, minlength=4) * weights
    torch.testing.assert_close(weighted_counts[:3], torch.tensor([2.0, 2.0, 2.0]))


def test_weighted_cross_entropy_matches_torch() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 0.5]])
    labels = torch.tensor([0, 1, 1])
    weights = torch.tensor([0.75, 1.5])
    actual = behavior_cloning_cross_entropy(
        logits, labels, class_weights=weights
    )
    expected = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
    torch.testing.assert_close(actual, expected)


def test_classification_metrics_include_balanced_accuracy_f1_and_confusion() -> None:
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    predictions = torch.tensor([0, 1, 1, 1, 0, 2])
    metrics = classification_metrics(predictions, targets, num_classes=3)

    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.balanced_accuracy == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert metrics.macro_f1 == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    torch.testing.assert_close(
        metrics.confusion_counts,
        torch.tensor([[1, 1, 0], [0, 2, 0], [1, 0, 1]]),
    )


def test_behavior_cloning_training_learns_phase_classifier() -> None:
    torch.manual_seed(11)
    dataset = _dataset()
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    initial = evaluate_behavior_cloning(model, dataset)
    result = train_behavior_cloning(
        model,
        dataset,
        optimizer,
        epochs=80,
        batch_size=3,
        class_weights=inverse_frequency_class_weights(
            dataset.phase_labels, num_classes=3
        ),
    )
    trained = evaluate_behavior_cloning(model, dataset)

    assert len(result.losses) == 80
    assert result.final_loss < result.losses[0]
    assert trained.accuracy == pytest.approx(1.0)
    assert trained.balanced_accuracy == pytest.approx(1.0)
    assert trained.macro_f1 == pytest.approx(1.0)
    assert trained.accuracy >= initial.accuracy


def test_behavior_cloning_can_train_and_evaluate_option_labels() -> None:
    dataset = HarvestImitationDataset.from_option_labels(
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 3.0]]),
        torch.tensor([0, 1, 0, 1]),
        ("HOLD_HIGH", "HARVEST_UNDERCUT"),
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    train_behavior_cloning(
        model, dataset, optimizer, target="option", epochs=50
    )
    metrics = evaluate_behavior_cloning(model, dataset, target="option")
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.confusion_counts.shape == (2, 2)


@pytest.mark.parametrize(
    "train_fraction",
    [0.0, 1.0, -0.1, 1.1],
)
def test_temporal_split_rejects_invalid_fraction(train_fraction: float) -> None:
    with pytest.raises(ValueError, match="train_fraction"):
        temporal_train_validation_split(_dataset(), train_fraction=train_fraction)
