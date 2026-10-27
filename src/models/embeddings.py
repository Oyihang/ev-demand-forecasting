
from typing import List, Optional
from torch import nn, Tensor
import torch

class ContinuousEmbeddings(nn.Module):
    """
    Provides embeddings for each continuous variable according to TFT paper.
    Each variable will be independently linearly projected to a hidden_dim space.

    Args:
        hidden_dim: int
            dimension to which each input is projected.
        num_continuous_variables: int
            number of variables that needs to be projected.
    """
    def __init__(self, hidden_dim: int, num_continuous_variables: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_continuous_variables = num_continuous_variables
        self.embedding_layers = nn.ModuleList(
            [nn.Linear(1, hidden_dim) for _ in range(num_continuous_variables)]
        )
    
    def forward(self, x: Tensor):
        """
        x: tensor containing all the continuous variables concatenated along the last axis 
        x shape = (*, C)
        """
        C = x.shape[-1]     # x shape = (B, T, C)
        assert self.num_continuous_variables == C
        out_emb = []
        for i in range(C):
            input_variable = x[..., i : i + 1]  # (*, 1)，只保留第 i 个变量
            out_emb.append(self.embedding_layers[i](input_variable))  # (*, H)，对每个变量单独进行线性映射
        return torch.stack(out_emb, dim=-2)


class CategoricalEmbeddings(nn.Module):
    """
    def __init__(self, hidden_dim: int, num_continuous_variables: int):
    Each variable will be independently projected to a hidden_dim space.

    Args:
        hidden_dim: int
            dimension to which each input is projected.
        num_categorical_variables: int
            number of variables that needs to be projected.
        num_categories_per_variable: list of int
            list containing the number of categories for each categorical variable.
    """
    def __init__(self, hidden_dim: int, category_counts: List[int]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.category_counts = category_counts
        self.embed_lookup_table = nn.ModuleList(
            [nn.Embedding(cat_size, hidden_dim) for cat_size in category_counts]
        )

    def forward(self, x: Tensor):
        """
        x: tensor containing all the continuous variables concatenated along the last axis 
        x shape = (*, C)
        """
        C = x.shape[-1]
        assert len(self.category_counts) == C
        out_emb = []
        for i in range(C):
            input_variable = x[..., i : i + 1]  # (*, 1)
            assert torch.all(input_variable < self.category_counts[i]), (
                f"Check category_counts = {self.category_counts}"
            )
            out_emb.append(self.embed_lookup_table[i](input_variable.long()))  # (*, 1, H)
        return torch.cat(out_emb, dim=-2)


class TFTEmbeddings(nn.Module):
    """
    Embedding module for the Temporal Fusion Transformer (TFT).

    This module transforms raw time series inputs into learned embeddings, as defined in TFT paper:
    - Each continuous variable is projected via its own linear layer
    - Each categorical variable is embedded via its own embedding table
    - Inputs are split into semantic groups: known, observed, and static variables

    Args:
        hidden_dim : int
            Dimensionality of the embedding space used throughout the TFT.
        known_inputs : List[str], optional
            Names of time-varying inputs that are known for all timesteps
            (past and future), e.g. calendar features, planned promotions.
        known_inputs_categorical_cardinalities : List[int], optional
            Cardinalities of categorical variables among `known_inputs`.
            The order must match the order of categorical variables as they
            appear at the end of the input tensor.
        observed_inputs : List[str], optional
            Names of time-varying inputs that are only observed historically
            (not known in the future), e.g. target variable or lagged signals.
        observed_inputs_categorical_cardinalities : List[int], optional
            Cardinalities of categorical variables among `observed_inputs`.
            Same ordering constraint as above applies.
        static_covariates : List[str], optional
            Names of static (time-invariant) variables, e.g. entity identifiers
            such as store ID, product ID, etc.
        static_covariates_categorical_cardinalities : List[int], optional
            Cardinalities of categorical variables among `static_covariates`.

    Input Format:
        Each input group (known, observed, static) must be provided as a tensor
        where:
            - Continuous variables come first
            - Categorical variables are concatenated at the end
            - Variable order matches the corresponding name list

                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_continuous_variables = num_continuous_variables
                self.embedding_layers = nn.ModuleList(
                    [nn.Linear(1, hidden_dim) for _ in range(num_continuous_variables)]
                )
            known_inputs = ["var1", "var2", "var3", "var4", "var5"]
            known_inputs_categorical_cardinalities = [7, 4]

        implies:
            - var1, var2, var3 are continuous
            - var4 is categorical and has cardinality 7
            - var5 is categorical and has cardinality 4

        The corresponding tensor must have shape:
            (batch_size, time_steps, 5)

        with variables ordered exactly as specified in `known_inputs`.

    Returns:
        The module typically produces embedded tensors grouped as:
            - static embeddings: (B, N_static, hidden_dim)
            - observed embeddings: (B, T, N_observed, hidden_dim)
            - known embeddings: (B, T, N_known, hidden_dim)

    Warning: Correct ordering of variables and categorical cardinalities is essential to avoid silent misalignment bugs.
    """
    def __init__(
        self,
        hidden_dim: int,
        known_inputs: List[str],
        known_inputs_categorical_cardinalities: List[int],
        observed_inputs: List[str],
        observed_inputs_categorical_cardinalities: List[int],
        static_covariates: List[str],
        static_covariates_categorical_cardinalities: List[int],
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.known_inputs = known_inputs
        self.known_inputs_categorical_cardinalities = known_inputs_categorical_cardinalities
        self.observed_inputs = observed_inputs
        self.observed_inputs_categorical_cardinalities = observed_inputs_categorical_cardinalities
        self.static_covariates = static_covariates
        self.static_covariates_categorical_cardinalities = static_covariates_categorical_cardinalities
        self.num_known_inputs_continuous = len(known_inputs) - len(known_inputs_categorical_cardinalities)  # 必须按照顺序，[连续变量，类别变量]
        self.num_observed_inputs_continuous = len(observed_inputs) - len(observed_inputs_categorical_cardinalities)
        self.num_static_inputs_continuous = len(static_covariates) - len(static_covariates_categorical_cardinalities)

        # Input embedding and transformations
        self.known_inputs_cont_emb, self.known_inputs_cat_emb = self.get_continous_and_categorical_embeddings(
            known_inputs,
            known_inputs_categorical_cardinalities,
            hidden_dim,
        )

        self.observed_inputs_cont_emb, self.observed_inputs_cat_emb = self.get_continous_and_categorical_embeddings(
            observed_inputs,
            observed_inputs_categorical_cardinalities,
            hidden_dim,
        )

        self.static_covariates_cont_emb, self.static_covariates_cat_emb = self.get_continous_and_categorical_embeddings(
            static_covariates,
            static_covariates_categorical_cardinalities,
            hidden_dim,
        )
    
    @staticmethod
    def get_continous_and_categorical_embeddings(
        input_variable_names: List[str],    # 变量名
        categorical_cardinalities: List[int],   # 类别变量的基数
        embedding_dim: int,
    ):
        total_num_inputs = len(input_variable_names)
        num_inputs_categorical = len(categorical_cardinalities)
        num_inputs_continuous = total_num_inputs - num_inputs_categorical

        if num_inputs_continuous:
            continuous_emb = ContinuousEmbeddings(embedding_dim, num_inputs_continuous)
        else:
            continuous_emb = None

        if num_inputs_categorical:
            categorical_emb = CategoricalEmbeddings(embedding_dim, categorical_cardinalities)
        else:
            categorical_emb = None

        return continuous_emb, categorical_emb
    
    def _validate_input_tensor(
        self,
        x: Tensor,
        input_variable_names: List[str],
        categorical_cardinalities: List[int],
        group_name: str,
        expect_time_dim: bool = True,
    ):
        if x is None:
            return

        if not isinstance(x, Tensor):
            raise TypeError(f"{group_name} input must be a torch.Tensor")

        if expect_time_dim: # 用于判断 known 和 observed 输入是否有时间维度
            if x.ndim != 3: 
                raise ValueError(
                    f"{group_name} must have shape (B, T, N), got {x.shape}"
                )
            B, T, N = x.shape
        else:
            if x.ndim != 2:
                raise ValueError(f"{group_name} must have shape (B, N), got {x.shape}")
            B, N = x.shape

        if N != len(input_variable_names):
            raise ValueError(
                f"{group_name}: expected {len(input_variable_names)} variables, got {N}"
            )

        num_cat = len(categorical_cardinalities)
        num_cont = N - num_cat

        # Check tensor type
        if num_cont > 0:
            assert torch.is_floating_point(x), (
                f"{num_cont} continuous variables are present in tensor, its dtype must be float, got {x.dtype} for {group_name}"
            )
        else:
            assert x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64), (
                f"Only categorical variables are present in tensor, its dtype must be integer, got {x.dtype} for {group_name}"
            )

        # Categorical part checks
        if num_cat > 0:
            cat_slice = x[..., num_cont:]

            # Check integer values
            if torch.is_floating_point(cat_slice):
                if not torch.all(cat_slice == cat_slice.long()):
                    raise ValueError(
                        f"{group_name}: categorical values must be integers"
                    )
                cat_slice = cat_slice.long()

            # Check bounds
            for i, cardinality in enumerate(categorical_cardinalities):
                values = cat_slice[..., i]
                if values.min() < 0 or values.max() >= cardinality:
                    raise ValueError(
                        f"{group_name}: categorical variable {i} has values outside [0, {cardinality - 1}]"
                    )
    
    def validate_inputs(
        self,
        known_inputs: Tensor,
        observed_inputs: Tensor,
        static_inputs: Optional[Tensor] = None,
    ):
        """
        Validates input tensors against the embedding configuration.

        This method should be called at the beginning of `forward()` to ensure:
        - correct shapes
        - correct dtype (float for continuous, int for categorical)
        - correct cardinality bounds for categorical variables

        Raises TypeError or ValueError if any inconsistency is detected.
        """

        self._validate_input_tensor(
            known_inputs,
            self.known_inputs,
            self.known_inputs_categorical_cardinalities,
            "known_inputs",
            expect_time_dim=True,
        )

        self._validate_input_tensor(
            observed_inputs,
            self.observed_inputs,
            self.observed_inputs_categorical_cardinalities,
            "observed_inputs",
            expect_time_dim=True,
        )

        self._validate_input_tensor(
            static_inputs,
            self.static_covariates,
            self.static_covariates_categorical_cardinalities,
            "static_covariates",
            expect_time_dim=False,
        )
    
    def forward(
        self,
        observed_inputs: Tensor,
        known_inputs: Tensor,
        static_covariates: Optional[Tensor] = None,
    ):
        """
        Args:
            observed_inputs: tensor of shape (B, context_length, num_observed_inputs) -> The first channel corresponds to the past target
            known_inputs: tensor of shape (B, context_length + horizon_length, num_known_inputs)
            static_covariates: tensor of shape (B, num_static)

        Outputs:
            observed_inputs_embedded: tensor of shape (B, context_length, num_observed_inputs, hidden_dim)
            known_inputs_embedded: tensor of shape (B, context_length + horizon_length, num_known_inputs, hidden_dim)
            static_covariates_embedded: tensor of shape (B, num_static, hidden_dim)
        """
        self.validate_inputs(
            observed_inputs=observed_inputs,
            known_inputs=known_inputs,
            static_inputs=static_covariates,
        )

        # observed只有历史时间步
        observed_inputs_embedded = []
        if self.num_observed_inputs_continuous > 0:
            cont_slice = observed_inputs[:, :, : self.num_observed_inputs_continuous]
            observed_inputs_embedded.append(self.observed_inputs_cont_emb(cont_slice))
        if len(self.observed_inputs_categorical_cardinalities) > 0:
            cat_slice = observed_inputs[:, :, self.num_observed_inputs_continuous :].long()  # nn.embeddings expects long tensors
            observed_inputs_embedded.append(self.observed_inputs_cat_emb(cat_slice))
        observed_inputs_embedded = torch.cat(observed_inputs_embedded, dim=-2)  # (B, context_length, num_observed_inputs, hidden_dim)

        # known有历史时间步和未来时间步
        known_inputs_embedded = []
        if self.num_known_inputs_continuous > 0:
            cont_slice = known_inputs[:, :, : self.num_known_inputs_continuous]
            known_inputs_embedded.append(self.known_inputs_cont_emb(cont_slice))
        if len(self.known_inputs_categorical_cardinalities) > 0:
            cat_slice = known_inputs[:, :, self.num_known_inputs_continuous :].long()
            known_inputs_embedded.append(self.known_inputs_cat_emb(cat_slice))
        known_inputs_embedded = torch.cat(known_inputs_embedded, dim=-2)  # (B, context_length + horizon_length, num_known_inputs, hidden_dim)

        if (self.static_covariates_cont_emb is not None) or (self.static_covariates_cat_emb is not None):
            # The class expects static covariates as inputs
            assert static_covariates is not None, (
                f"The model expects static covariates {self.static_covariates} but no tensor was passed to forward"
            )
            static_covariates_embedded = []
            if self.num_static_inputs_continuous > 0:
                cont_slice = static_covariates[:, : self.num_static_inputs_continuous]
                static_covariates_embedded.append(self.static_covariates_cont_emb(cont_slice))
            if len(self.static_covariates_categorical_cardinalities) > 0:
                cat_slice = static_covariates[:, self.num_static_inputs_continuous :].long()
                static_covariates_embedded.append(self.static_covariates_cat_emb(cat_slice))
            static_covariates_embedded = torch.cat(static_covariates_embedded, dim=-2)  # (B, num_static, hidden_dim)
        else:
            static_covariates_embedded = None

        return observed_inputs_embedded, known_inputs_embedded, static_covariates_embedded
