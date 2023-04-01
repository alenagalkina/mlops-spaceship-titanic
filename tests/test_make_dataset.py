import great_expectations as ge
import pandas as pd


def test_output():
    df = pd.read_csv("data/interim/train_postproc.csv")
    df_ge = ge.from_pandas(df)

    # expected_columns = []
    # assert df_ge.expect_table_columns_to_match_ordered_list(column_list=expected_columns).success is True
    # assert df_ge.expect_column_values_to_be_unique(column="id").success is True is True
    assert (
        df_ge.expect_column_values_to_not_be_null(column="Transported").success
        is True
        is True
    )
    assert (
        df_ge.expect_column_values_to_be_of_type(column="Age", type_="float64").success
        is True
    )
