from models.tft import TemporalFusionTransformer

import torch 

def test_forward():
    """Run the model forward on a dummy example."""
    context_length, horizon_length = 168, 24
    batch_size = 8

    hidden_dim = 128  # d_model
    quantiles = (0.1, 0.5, 0.9)

    # observed inputs are time varying inputs for which we only know the historical values
    # e.g. target variable and potential exogenous covariates, for which we don't know future values a priori
    num_observed_inputs_continuous = 2  # We have 2 continuous observed inputs (only past values are known)
    observed_inputs_categorical_cardinalities = [
        12,
        24,
        192,
    ]  # We have 3 categorical observed inputs, and their respective cardinalities are 12, 24, and 192
    # known inputs are time varying inputs for which values are known for all timesteps (past and future)
    # e.g. calendar features, planned events, or potential exogenous variables.
    num_known_inputs_continuous = 3  # We have 3 continuous known inputs (past + future values are known)
    known_inputs_categorical_cardinalities = [
        2,
        8,
    ]  # We have 2 categorical known inputs, and their respective cardinalities are 2, and 8
    # static covariates don't vary with time.
    # e.g., store id or city in the Retail dataset
    num_static_covariates_continuous = 1  # We have one continuous static covariate
    static_covariates_categorical_cardinalities = [
        3,
    ]  # We have one categorical static covariate, and its cardinality is 3 -> its values are integer between 0 and 2 (included).

    # We build dummy inputs to give the reader
    # The categorical inputs must be given to the model in the exact same order as the one provided in the cardinality list (e.g., known_inputs_categorical_cardinalities for known inputs)
    known_inputs_cont = torch.randn(batch_size, context_length + horizon_length, num_known_inputs_continuous)
    known_inputs_cat = torch.cat(
        [
            torch.randint(0, cardinality, size=(batch_size, context_length + horizon_length, 1))
            for cardinality in known_inputs_categorical_cardinalities
        ],
        dim=-1,
    )
    observed_inputs_cont = torch.randn(batch_size, context_length, num_observed_inputs_continuous)
    observed_inputs_cat = torch.cat(
        [
            torch.randint(0, cardinality, size=(batch_size, context_length, 1))
            for cardinality in observed_inputs_categorical_cardinalities
        ],
        dim=-1,
    )
    static_covariates_cont = torch.randn(batch_size, num_static_covariates_continuous)
    static_covariates_cat = torch.cat(
        [
            torch.randint(0, cardinality, size=(batch_size, 1))
            for cardinality in static_covariates_categorical_cardinalities
        ],
        dim=-1,
    )

    # When building known_inputs, observed_inputs, and static_covariates tensors, continuous variables should always come first along the feature dimension
    known_inputs = torch.cat([known_inputs_cont, known_inputs_cat], dim=-1)
    observed_inputs = torch.cat([observed_inputs_cont, observed_inputs_cat], dim=-1)
    static_covariates = torch.cat([static_covariates_cont, static_covariates_cat], dim=-1)

    # The name of the inputs must match the order of the inputs in the tensors along the feature dimension
    known_inputs_names = [f"known_{i}" for i in range(known_inputs.shape[-1])]
    observed_inputs_names = [f"observed_{i}" for i in range(observed_inputs.shape[-1])]
    static_covariates_names = [f"static_{i}" for i in range(static_covariates.shape[-1])]

    # instanciate model
    tft = TemporalFusionTransformer(
        horizon_length=horizon_length,
        hidden_dim=hidden_dim,
        known_inputs=known_inputs_names,
        observed_inputs=observed_inputs_names,
        static_covariates=static_covariates_names,
        known_inputs_categorical_cardinalities=known_inputs_categorical_cardinalities,
        observed_inputs_categorical_cardinalities=observed_inputs_categorical_cardinalities,
        static_covariates_categorical_cardinalities=static_covariates_categorical_cardinalities,
        num_heads=1,
        dropout=0.0,
        lstm_layers=1,
        quantiles=quantiles,
    )

    # run forward
    out = tft(
        observed_inputs=observed_inputs,
        known_inputs=known_inputs,
        static_covariates=static_covariates,
    )
    print(f"Output shape: {out.shape}")

    assert out.shape == (batch_size, horizon_length, len(quantiles))


if __name__ == "__main__":
    test_forward()
