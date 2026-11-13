import re

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


COLUMN_NAMES = [
    "URL",
    "Product name",
    "Current price",
    "Category",
    "Weight",
    "Material",
    "Gender",
    "Rim material",
    "Width",
    "Axle dimension",
]

FRAME_ONLY_PATTERNS = [
    re.compile(r"\bframeset\b", re.IGNORECASE),
    re.compile(r"\bframe\s+kit\b", re.IGNORECASE),
    re.compile(r"\bframe\s+and\s+brake\s+kit\b", re.IGNORECASE),
]


def load_data():
    return pd.read_csv(
        "data.tsv",
        sep="\t",
        header=None,
        names=COLUMN_NAMES,
        encoding="utf-8-sig",
    )


def is_potentially_incomplete(product_name):
    if pd.isna(product_name):
        return False
    return any(pattern.search(str(product_name)) for pattern in FRAME_ONLY_PATTERNS)


def parse_price(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    text = re.sub(r"^from\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    number_text = re.sub(r"Ç\s*$", "", text).strip()
    if not re.fullmatch(r"\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:,\d+)?", number_text):
        return None
    return float(number_text.replace(".", "").replace(",", "."))


def parse_weight(value):
    if pd.isna(value):
        return None

    match = re.fullmatch(r"\s*(\d+(?:[,.]\d+)?)\s*kg\s*", str(value), re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1).replace(",", "."))


def parse_width(value):
    if pd.isna(value):
        return None

    text = str(value).strip().replace(",", ".")
    millimetres = re.fullmatch(r"(\d+(?:\.\d+)?)\s*mm", text, re.IGNORECASE)
    if millimetres:
        return float(millimetres.group(1))

    inches = re.fullmatch(r'(\d+(?:\.\d+)?)\s*"', text)
    if inches:
        return float(inches.group(1)) * 25.4
    return None


def report_parse_failures(original, parsed, column):
    failures = original[column].notna() & parsed.isna()
    print(f"\n{column} parsing failures: {failures.sum()}")
    if failures.any():
        print(f"  Unexpected values: {original.loc[failures, column].unique().tolist()}")


def prepare_working_data(data):
    frame_only_mask = data["Product name"].map(is_potentially_incomplete)
    frame_only_rows = data[frame_only_mask]
    remaining = data.loc[~frame_only_mask].copy()

    audit_columns = ["URL", "Product name"]
    audit_table = remaining[audit_columns].copy()
    working = remaining.drop(columns=audit_columns).copy()
    working["Current price (EUR)"] = working["Current price"].map(parse_price)
    working["Weight (kg)"] = working["Weight"].map(parse_weight)
    working["Width (mm)"] = working["Width"].map(parse_width)
    working = working.drop(columns=["Current price", "Weight", "Width"])

    categorical_columns = [
        "Category",
        "Material",
        "Gender",
        "Rim material",
        "Axle dimension",
    ]
    for column in categorical_columns:
        working[column] = working[column].astype("category")

    return frame_only_rows, audit_table, working


def print_rows(title, rows):
    print(f"\n{title}")
    if rows.empty:
        print("  None")
        return

    selected_columns = [
        "Product name",
        "Category",
        "Current price",
        "Weight",
        "URL",
    ]
    print(rows[selected_columns].to_string(index=False))


def print_numeric_summaries(working):
    print("\nNumeric summaries:")
    print(
        working[["Current price (EUR)", "Weight (kg)", "Width (mm)"]]
        .describe()
        .to_string()
    )


def deterministic_mode(values):
    """Return the alphabetically first value among the most frequent values."""
    counts = values.dropna().value_counts()
    if counts.empty:
        return None
    highest_count = counts.max()
    candidates = [value for value, count in counts.items() if count == highest_count]
    return sorted(candidates, key=str)[0]


def impute_missing_values(working):
    """Impute missing values on a copy, using category-aware fallbacks."""
    imputed = working.copy()
    numeric_imputations = []

    for column in ["Weight (kg)", "Width (mm)"]:
        global_median = imputed[column].median()
        for index in imputed.index[imputed[column].isna()]:
            category = imputed.at[index, "Category"]
            category_values = imputed.loc[
                imputed["Category"] == category, column
            ].dropna()
            replacement = (
                category_values.median()
                if not category_values.empty
                else global_median
            )
            imputed.at[index, column] = replacement
            numeric_imputations.append(
                (column, index, category, replacement)
            )

    categorical_imputations = []
    for column in ["Gender", "Rim material"]:
        if "Unknown" not in imputed[column].cat.categories:
            imputed[column] = imputed[column].cat.add_categories(["Unknown"])
        missing_indices = imputed.index[imputed[column].isna()]
        imputed.loc[missing_indices, column] = "Unknown"
        categorical_imputations.append((column, len(missing_indices), "Unknown"))

    global_mode = deterministic_mode(imputed["Axle dimension"])
    for index in imputed.index[imputed["Axle dimension"].isna()]:
        category = imputed.at[index, "Category"]
        category_mode = deterministic_mode(
            imputed.loc[
                imputed["Category"] == category, "Axle dimension"
            ]
        )
        replacement = category_mode if category_mode is not None else global_mode
        imputed.at[index, "Axle dimension"] = replacement
        categorical_imputations.append(
            ("Axle dimension", 1, replacement)
        )

    print("\nNumeric imputations:")
    for column, index, category, replacement in numeric_imputations:
        print(
            f"  {column}: row {index}, category {category!r}, "
            f"value {replacement}"
        )

    print("\nCategorical imputations:")
    for column, count, replacement in categorical_imputations:
        print(f"  {column}: {count} value(s) filled with {replacement!r}")

    return imputed


def discretize_numeric_column(data, source_column, target_column, labels):
    """Create three ordered quantile classes and report their boundaries."""
    quantiles = data[source_column].quantile([0, 1 / 3, 2 / 3, 1]).tolist()
    boundaries = sorted(set(quantiles))
    print(f"\n{target_column} quantile boundaries: {quantiles}")

    if len(boundaries) != len(labels) + 1:
        raise ValueError(
            f"Cannot create {len(labels)} non-empty classes for {source_column}: "
            f"duplicated quantile boundaries {quantiles}"
        )

    result = pd.cut(
        data[source_column],
        bins=boundaries,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )
    frequencies = result.value_counts(sort=False)
    if (frequencies == 0).any() or result.isna().any():
        raise ValueError(
            f"Discretization produced an empty class or missing value for "
            f"{source_column}: {frequencies.to_dict()}"
        )

    print(f"{target_column} frequencies:")
    print(frequencies.to_string())
    return result


def create_categorical_dataset(imputed_working, output_path="categorical.csv"):
    """Create and validate the categorical mining dataset."""
    categorical = imputed_working.copy()
    categorical["Price class"] = discretize_numeric_column(
        categorical,
        "Current price (EUR)",
        "Price class",
        ["Low", "Medium", "High"],
    )
    categorical["Weight class"] = discretize_numeric_column(
        categorical,
        "Weight (kg)",
        "Weight class",
        ["Light", "Medium", "Heavy"],
    )
    categorical["Width class"] = discretize_numeric_column(
        categorical,
        "Width (mm)",
        "Width class",
        ["Narrow", "Medium", "Wide"],
    )
    categorical = categorical.drop(
        columns=["Current price (EUR)", "Weight (kg)", "Width (mm)"]
    )
    output_columns = [
        "Category",
        "Material",
        "Gender",
        "Rim material",
        "Axle dimension",
        "Price class",
        "Weight class",
        "Width class",
    ]
    categorical = categorical[output_columns]

    assert categorical.shape == (157, 8)
    assert not categorical.isna().any().any()
    assert len(categorical.select_dtypes(include="number").columns) == 0
    categorical.to_csv(output_path, sep=",", encoding="utf-8", index=False)
    return categorical


def create_numerical_dataset(imputed_working, output_path="numerical.csv"):
    """Create and validate the numeric regression dataset."""
    numerical = imputed_working.copy()
    target_column = "Current price (EUR)"
    numeric_features = ["Weight (kg)", "Width (mm)"]
    categorical_features = [
        "Category",
        "Material",
        "Gender",
        "Rim material",
        "Axle dimension",
    ]

    scaler = MinMaxScaler()
    numerical[numeric_features] = scaler.fit_transform(numerical[numeric_features])
    print("\nMin-Max scaling ranges:")
    for column, minimum, maximum in zip(
        numeric_features, scaler.data_min_, scaler.data_max_
    ):
        print(f"  {column}: original min={minimum}, original max={maximum}")

    encoded = pd.get_dummies(
        numerical[categorical_features],
        columns=categorical_features,
        prefix=categorical_features,
        prefix_sep="_",
        dtype=float,
    )
    encoded = encoded.reindex(sorted(encoded.columns), axis=1)
    print("\nOne-hot columns generated:")
    for column in categorical_features:
        count = sum(name.startswith(f"{column}_") for name in encoded.columns)
        print(f"  {column}: {count}")

    numerical = pd.concat(
        [
            numerical[[target_column] + numeric_features],
            encoded,
        ],
        axis=1,
    )
    assert numerical.shape[0] == 157
    assert not numerical.isna().any().any()
    assert all(
        pd.api.types.is_numeric_dtype(numerical[column])
        for column in numerical.columns
    )
    feature_columns = [column for column in numerical.columns if column != target_column]
    assert (
        numerical[feature_columns].ge(0).all().all()
        and numerical[feature_columns].le(1).all().all()
    )
    numerical.to_csv(output_path, sep=",", encoding="utf-8", index=False)
    return numerical


def main():
    data = load_data()
    frame_only_rows, audit_table, working = prepare_working_data(data)

    print("Excluded frame-only records:")
    print_rows("", frame_only_rows)
    print(f"\nExcluded rows: {len(frame_only_rows)}")
    print(
        f"Distinct excluded product names: "
        f"{frame_only_rows['Product name'].nunique()}"
    )

    parsed_weights = data["Weight"].map(parse_weight)
    low_weight_rows = data[parsed_weights.lt(5).fillna(False)]
    print_rows("Products with weight below 5 kg:", low_weight_rows)
    print(f"\nProducts below 5 kg: {len(low_weight_rows)}")

    report_parse_failures(
        data.loc[~data.index.isin(frame_only_rows.index)],
        working["Current price (EUR)"],
        "Current price",
    )
    report_parse_failures(
        data.loc[~data.index.isin(frame_only_rows.index)],
        working["Weight (kg)"],
        "Weight",
    )
    report_parse_failures(
        data.loc[~data.index.isin(frame_only_rows.index)],
        working["Width (mm)"],
        "Width",
    )

    print("\nWorking dataset:")
    print(f"  Rows: {working.shape[0]}")
    print(f"  Columns: {working.shape[1]}")
    print("\nColumn dtypes:")
    print(working.dtypes)
    print_numeric_summaries(working)
    print("\nMissing values per working column:")
    print(working.isna().sum())
    print(f"\nRows with at least one missing value: {working.isna().any(axis=1).sum()}")

    print("\nMissing values before imputation:")
    print(working.isna().sum())
    imputed_working = impute_missing_values(working)
    print("\nMissing values after imputation:")
    print(imputed_working.isna().sum())

    preserved_columns = ["Category", "Material", "Current price (EUR)"]
    for column in preserved_columns:
        assert imputed_working[column].equals(working[column])
    assert len(imputed_working) == 157
    assert not imputed_working.isna().any().any()
    print("\nImputation validation:")
    print(f"  Exactly 157 rows remain: {len(imputed_working) == 157}")
    print("  No missing values remain in the eight working columns: True")
    print("  Category, Material and Current price (EUR) unchanged: True")

    print("\nValidation:")
    print(f"  Exactly 157 rows remain: {working.shape[0] == 157}")
    print(f"  All remaining prices are valid: {working['Current price (EUR)'].notna().all()}")
    print(f"  Original row count is unchanged in memory: {len(data) == 160}")
    print("  data.tsv was only read and was not modified.")

    categorical = create_categorical_dataset(imputed_working)
    print("\nCategorical dataset validation:")
    print(f"  Shape: {categorical.shape}")
    print(f"  Missing values: {categorical.isna().sum().sum()}")
    print(
        f"  Numeric columns: "
        f"{len(categorical.select_dtypes(include='number').columns)}"
    )
    print("\nFirst five rows of categorical.csv:")
    print(categorical.head().to_string(index=False))
    print("\nOutput file: categorical.csv")

    numerical = create_numerical_dataset(imputed_working)
    print("\nNumerical dataset validation:")
    print(f"  Shape: {numerical.shape}")
    print(f"  Missing values: {numerical.isna().sum().sum()}")
    print(
        f"  All columns numeric: "
        f"{all(pd.api.types.is_numeric_dtype(dtype) for dtype in numerical.dtypes)}"
    )
    print(
        f"  All feature values in [0, 1]: "
        f"{numerical.drop(columns=['Current price (EUR)']).ge(0).all().all() and numerical.drop(columns=['Current price (EUR)']).le(1).all().all()}"
    )
    print("\nFirst five rows of numerical.csv:")
    print(numerical.head().to_string(index=False))
    print("\nOutput file: numerical.csv")


if __name__ == "__main__":
    main()
