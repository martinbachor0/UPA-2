import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

SEMANTIC_TYPES = {
    "URL": "identifier",
    "Product name": "text",
    "Current price": "numeric stored as text",
    "Category": "categorical",
    "Weight": "numeric stored as text",
    "Material": "categorical",
    "Gender": "categorical",
    "Rim material": "categorical",
    "Width": "numeric stored as text",
    "Axle dimension": "text",
}


def parse_price(value):
    """Parse a displayed EUR price, returning None for invalid input."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    text = re.sub(r"^from\s+", "", text, flags=re.IGNORECASE)
    text = text.replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)

    # The source contains several possible currency/display characters.
    text_without_currency = re.sub(r"(?:€|Ç|\ufffd)\s*$", "", text).strip()
    if text_without_currency == text:
        return None
    if not re.fullmatch(r"\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:,\d+)?", text_without_currency):
        return None

    number = text_without_currency.replace(".", "").replace(",", ".")
    return float(number)


def parse_weight(value):
    """Parse a weight in kilograms, returning None for invalid input."""
    if pd.isna(value):
        return None

    match = re.fullmatch(r"\s*(\d+(?:[,.]\d+)?)\s*kg\s*", str(value), flags=re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1).replace(",", "."))


def parse_width(value):
    """Parse a tyre width in millimetres, converting inch values."""
    if pd.isna(value):
        return None

    text = str(value).strip().lower().replace(",", ".")
    millimetres = re.fullmatch(r"(\d+(?:\.\d+)?)\s*mm", text)
    if millimetres:
        return float(millimetres.group(1))

    inches = re.fullmatch(r"(\d+(?:\.\d+)?)\s*\"", text)
    if inches:
        return float(inches.group(1)) * 25.4
    return None


def print_attribute_analysis(data):
    print("\nAnalysis of individual attributes:")
    for column in data.columns:
        values = data[column].dropna()
        print(f"\n{column} ({SEMANTIC_TYPES[column]}):")
        print(f"  Non-missing: {values.size}")
        print(f"  Missing: {data[column].isna().sum()}")
        print(f"  Unique non-missing: {values.nunique()}")
        print("  Most frequent values:")
        for value, count in values.value_counts().head(5).items():
            print(f"    {value!r}: {count}")


def print_numeric_summary(original, parsed, column, label):
    failures = original[column].notna() & parsed.isna()
    print(f"\n{label} analysis:")
    print(f"  Parsing failures: {failures.sum()}")
    if failures.any():
        print(f"  Failed values: {original.loc[failures, column].unique().tolist()}")

    valid = parsed.dropna()
    if valid.empty:
        print("  No valid numeric values.")
        return

    print(f"  Count: {valid.count()}")
    print(f"  Minimum: {valid.min()}")
    print(f"  Maximum: {valid.max()}")
    print(f"  Mean: {valid.mean()}")
    print(f"  Median: {valid.median()}")
    print(f"  Standard deviation: {valid.std()}")
    print(f"  Q1: {valid.quantile(0.25)}")
    print(f"  Q3: {valid.quantile(0.75)}")


def print_categorical_analysis(data, columns):
    print("\nDetailed categorical analysis:")
    for column in columns:
        counts = data[column].value_counts(dropna=False)
        percentages = counts / len(data) * 100
        table = pd.DataFrame(
            {
                "Count": counts,
                "Relative frequency (%)": percentages,
            }
        )

        print(f"\n{column}:")
        print(f"  Unique non-missing: {data[column].nunique(dropna=True)}")
        print(table)


def print_category_material_crosstabs(data):
    category = data["Category"].fillna("<missing>")
    material = data["Material"].fillna("<missing>")

    counts = pd.crosstab(category, material)
    percentages = pd.crosstab(category, material, normalize="index") * 100

    print("\nFrame material counts within each category:")
    print(counts)

    print("\nFrame material percentages within each category:")
    print(percentages)


def create_figures(analysis_data):
    figures_directory = Path("figures")
    figures_directory.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    price_data = analysis_data.dropna(subset=["Current price (EUR)"])
    plt.figure(figsize=(9, 6))
    sns.histplot(price_data["Current price (EUR)"], bins=18)
    plt.title("Distribution of Current Prices")
    plt.xlabel("Current price (EUR)")
    plt.ylabel("Number of bicycles")
    plt.tight_layout()
    plt.savefig(figures_directory / "01_price_histogram.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=price_data,
        x="Category",
        y="Current price (EUR)",
        order=sorted(price_data["Category"].dropna().unique()),
    )
    plt.title("Current Price by Bicycle Category")
    plt.xlabel("Category")
    plt.ylabel("Current price (EUR)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_directory / "02_price_by_category.png", dpi=150)
    plt.close()

    scatter_data = analysis_data.dropna(
        subset=["Weight (kg)", "Current price (EUR)", "Category"]
    )
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=scatter_data,
        x="Weight (kg)",
        y="Current price (EUR)",
        hue="Category",
        hue_order=sorted(scatter_data["Category"].unique()),
        s=70,
    )
    plt.title("Weight versus Current Price")
    plt.xlabel("Weight (kg)")
    plt.ylabel("Current price (EUR)")
    plt.legend(title="Category")
    plt.tight_layout()
    plt.savefig(figures_directory / "03_weight_vs_price.png", dpi=150)
    plt.close()

    material_by_category = pd.crosstab(
        analysis_data["Category"].fillna("<missing>"),
        analysis_data["Material"].fillna("<missing>"),
        normalize="index",
    ) * 100
    material_order = [
        material
        for material in [
            "Aluminium (AL)",
            "Carbon",
            "Carbon (CF)",
            "Carbon (CFR)",
            "<missing>",
        ]
        if material in material_by_category.columns
    ]
    material_by_category = material_by_category.reindex(
        columns=material_order, fill_value=0
    )
    ax = material_by_category.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        colormap="tab10",
    )
    ax.set_title("Frame Material Distribution by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Percentage within category (%)")
    ax.legend(title="Material", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_directory / "04_material_by_category.png", dpi=150)
    plt.close()

    weight_data = analysis_data.dropna(subset=["Weight (kg)", "Category"])
    weight_data = weight_data[weight_data["Category"] != "hybrid-city-bikes"]
    category_order = sorted(weight_data["Category"].unique())
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=weight_data,
        x="Category",
        y="Weight (kg)",
        order=category_order,
        inner="box",
    )
    plt.title(
        "Weight by Bicycle Category\n"
        "(hybrid-city-bikes excluded: only one product)"
    )
    plt.xlabel("Category")
    plt.ylabel("Weight (kg)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_directory / "05_weight_by_category.png", dpi=150)
    plt.close()

    price_by_material = analysis_data.dropna(
        subset=["Current price (EUR)", "Material"]
    )
    material_categories = [
        "Carbon (CF)",
        "Carbon (CFR)",
        "Aluminium (AL)",
        "Carbon",
    ]
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=price_by_material,
        x="Material",
        y="Current price (EUR)",
        order=material_categories,
    )
    plt.title(
        "Current Price by Frame Material\n"
        "(Carbon contains only one product)"
    )
    plt.xlabel("Frame material")
    plt.ylabel("Current price (EUR)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(figures_directory / "06_price_by_material.png", dpi=150)
    plt.close()


def print_outlier_record(row, value_column, method):
    print(
        f"    {method}: product={row['Product name']!r}, "
        f"category={row['Category']!r}, value={row[value_column]}"
    )


def print_outlier_analysis(analysis_data):
    """Report global and within-category IQR and Z-score outliers."""
    print("\nOutlier analysis:")
    numeric_columns = [
        ("Current price (EUR)", "Current price"),
        ("Weight (kg)", "Weight"),
    ]

    for value_column, label in numeric_columns:
        valid = analysis_data.dropna(subset=[value_column]).copy()
        values = valid[value_column]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        global_iqr = valid[
            (values < lower_bound) | (values > upper_bound)
        ]

        mean = values.mean()
        population_std = values.std(ddof=0)
        if population_std == 0:
            global_z = valid.iloc[0:0]
        else:
            z_scores = (values - mean) / population_std
            global_z = valid[z_scores.abs() > 3]

        iqr_indices = set(global_iqr.index)
        z_indices = set(global_z.index)
        print(f"\n{label} - global outliers:")
        print(f"  IQR Q1: {q1}, Q3: {q3}, IQR: {iqr}")
        print(f"  IQR bounds: [{lower_bound}, {upper_bound}]")
        print(f"  IQR count: {len(global_iqr)}")
        for _, row in global_iqr.iterrows():
            print_outlier_record(row, value_column, "IQR")

        print(f"  Z-score mean: {mean}, population standard deviation: {population_std}")
        print("  Z-score rule: absolute Z-score > 3")
        print(f"  Z-score count: {len(global_z)}")
        for _, row in global_z.iterrows():
            print_outlier_record(row, value_column, "Z-score")

        print(f"  Overlap between global methods: {len(iqr_indices & z_indices)}")

        print(f"\n{label} - within-category IQR outliers:")
        skipped_categories = []
        category_outliers = []
        for category, group in valid.groupby("Category", dropna=False):
            if len(group) < 4:
                skipped_categories.append((category, len(group)))
                continue

            group_values = group[value_column]
            group_q1 = group_values.quantile(0.25)
            group_q3 = group_values.quantile(0.75)
            group_iqr = group_q3 - group_q1
            group_lower = group_q1 - 1.5 * group_iqr
            group_upper = group_q3 + 1.5 * group_iqr
            detected = group[
                (group_values < group_lower) | (group_values > group_upper)
            ]
            category_outliers.extend(detected.index)
            print(
                f"  Category {category!r}: bounds "
                f"[{group_lower}, {group_upper}], count={len(detected)}"
            )
            for _, row in detected.iterrows():
                print_outlier_record(row, value_column, "within-category IQR")

        print(f"  Within-category IQR count: {len(category_outliers)}")
        if skipped_categories:
            print(f"  Skipped categories (< 4 valid observations): {skipped_categories}")
        else:
            print("  Skipped categories (< 4 valid observations): none")


def analyze_missing_values(data):
    """Describe missing values using only the original dataset columns."""
    original = data[COLUMN_NAMES]
    missing = original.isna()
    row_missing_counts = missing.sum(axis=1)
    total_cells = original.size
    total_missing = int(missing.sum().sum())

    print("\nDetailed missing-value analysis:")
    print(f"  Total missing cells: {total_missing} of {total_cells}")
    print(f"  Missing percentage of all cells: {total_missing / total_cells * 100:.2f}%")

    column_summary = pd.DataFrame(
        {
            "Missing count": missing.sum(),
            "Missing percentage (%)": missing.mean() * 100,
        }
    )
    print("\nMissing values by column:")
    print(column_summary)

    print("\nRows by number of missing attributes:")
    print(row_missing_counts.value_counts().sort_index().rename("Rows"))
    print(f"  Rows with zero missing values: {(row_missing_counts == 0).sum()}")
    print(f"  Rows with at least one missing value: {(row_missing_counts >= 1).sum()}")
    print(f"  Rows with more than one missing value: {(row_missing_counts > 1).sum()}")

    highest_count = row_missing_counts.max()
    highest_rows = original.loc[row_missing_counts == highest_count].copy()
    print(f"\nProducts with the highest number of missing attributes ({highest_count}):")
    for index, row in highest_rows.iterrows():
        missing_columns = list(missing.columns[missing.loc[index]])
        print(
            f"  Product: {row['Product name']!r}, "
            f"Category: {row['Category']!r}, URL: {row['URL']!r}, "
            f"Missing columns: {missing_columns}"
        )

    grouped_counts = missing.groupby(original["Category"], dropna=False).sum()
    grouped_percentages = (
        missing.groupby(original["Category"], dropna=False).mean() * 100
    )
    print("\nMissing-value counts grouped by Category:")
    print(grouped_counts.to_string())
    print("\nMissing-value percentages within each Category:")
    print(grouped_percentages.to_string())

    pair_columns = [
        "Gender",
        "Rim material",
        "Weight",
        "Material",
        "Width",
        "Axle dimension",
    ]
    print("\nPairs of attributes missing together:")
    for first, second in combinations(pair_columns, 2):
        pair_count = int((missing[first] & missing[second]).sum())
        print(f"  {first} + {second}: {pair_count}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        missing.astype(int),
        cmap=["white", "firebrick"],
        cbar=False,
        yticklabels=False,
        xticklabels=COLUMN_NAMES,
        linewidths=0.15,
        linecolor="lightgray",
    )
    plt.title("Missing-Value Matrix (red = missing)")
    plt.xlabel("Original attribute")
    plt.ylabel("Product row")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(Path("figures") / "07_missing_values.png", dpi=150)
    plt.close()


def analyze_correlations(analysis_data, original_data):
    """Calculate global and within-category correlations."""
    analysis_columns = ["Current price (EUR)", "Weight (kg)", "Width (mm)"]
    numeric_data = analysis_data[analysis_columns]

    print("\nCorrelation analysis:")
    print("\nWidth parsing:")
    width_failures = original_data["Width"].notna() & numeric_data["Width (mm)"].isna()
    print(f"  Parsing failures: {width_failures.sum()}")
    if width_failures.any():
        print(f"  Failed values: {original_data.loc[width_failures, 'Width'].unique().tolist()}")

    pearson = numeric_data.corr(method="pearson")
    spearman = numeric_data.corr(method="spearman")
    print("\nPearson correlation matrix:")
    print(pearson.round(3).to_string())
    print("\nSpearman correlation matrix:")
    print(spearman.round(3).to_string())

    print("\nComplete observation pairs:")
    for first, second in combinations(analysis_columns, 2):
        pair_count = numeric_data[[first, second]].dropna().shape[0]
        print(f"  {first} + {second}: {pair_count}")

    for method, matrix, filename in [
        ("Pearson", pearson, "08_pearson_correlation.png"),
        ("Spearman", spearman, "09_spearman_correlation.png"),
    ]:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            vmin=-1,
            vmax=1,
            center=0,
            cmap="coolwarm",
            square=True,
        )
        plt.title(f"{method} Correlations of Numeric Attributes")
        plt.xlabel("Attribute")
        plt.ylabel("Attribute")
        plt.tight_layout()
        plt.savefig(Path("figures") / filename, dpi=150)
        plt.close()

    print("\nPrice-weight correlations within categories:")
    category_data = analysis_data[["Category", "Current price (EUR)", "Weight (kg)"]]
    for category, group in category_data.groupby("Category", dropna=False):
        pairs = group.dropna(subset=["Current price (EUR)", "Weight (kg)"])
        if len(pairs) < 4:
            print(f"  Skipped {category!r}: only {len(pairs)} valid pairs")
            continue
        print(
            f"  {category!r}: n={len(pairs)}, "
            f"Pearson={pairs['Current price (EUR)'].corr(pairs['Weight (kg)'], method='pearson'):.3f}, "
            f"Spearman={pairs['Current price (EUR)'].corr(pairs['Weight (kg)'], method='spearman'):.3f}"
        )

    print(
        "\nInterpretation note: Correlation describes association and does not imply "
        "causation. Relationships across categories may differ from relationships "
        "within individual categories."
    )


def main():
    data = pd.read_csv(
        "data.tsv",
        sep="\t",
        header=None,
        names=COLUMN_NAMES,
        encoding="utf-8-sig",
    )

    print("Dataset shape:")
    print(data.shape)

    print("\nFirst five rows:")
    print(data.head())

    print("\nPandas dtypes:")
    print(data.dtypes)

    print("\nMissing-value counts:")
    print(data.isna().sum())

    print("\nRepr of the first three price values:")
    for value in data["Current price"].head(3):
        print(repr(value))

    print("\nUnicode code points of the first three original price values:")
    for value in data["Current price"].head(3):
        print(f"{value!r}: {[f'U+{ord(character):04X}' for character in value]}")

    print_attribute_analysis(data)

    # These are separate in-memory analysis columns; original text is preserved.
    analysis_data = data.copy()
    analysis_data["Current price (EUR)"] = data["Current price"].map(parse_price)
    analysis_data["Weight (kg)"] = data["Weight"].map(parse_weight)
    analysis_data["Width (mm)"] = data["Width"].map(parse_width)
    print_numeric_summary(
        data,
        analysis_data["Current price (EUR)"],
        "Current price",
        "Current price (EUR)",
    )
    print_numeric_summary(
        data,
        analysis_data["Weight (kg)"],
        "Weight",
        "Weight (kg)",
    )

    print("\nCategory counts:")
    print(data["Category"].value_counts())

    print("\nMaterial counts:")
    print(data["Material"].value_counts(dropna=False))

    categorical_columns = [
        "Category",
        "Material",
        "Gender",
        "Rim material",
        "Axle dimension",
    ]
    print_categorical_analysis(data, categorical_columns)
    print_category_material_crosstabs(data)
    create_figures(analysis_data)
    print_outlier_analysis(analysis_data)
    analyze_missing_values(data)
    analyze_correlations(analysis_data, data)


if __name__ == "__main__":
    main()
