# -*- coding: utf-8 -*-
import click
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def make_dataset(input_filepath: str, output_filepath: str):

    data = pd.read_csv(input_filepath)
    luxury_amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    data.loc[data.CryoSleep is True, luxury_amenities] = 0.0
    data.loc[data.CryoSleep is False, luxury_amenities] = data.loc[
        data.CryoSleep is False, luxury_amenities
    ].fillna(data.loc[data.CryoSleep is False, luxury_amenities].median())
    df = data.loc[data[luxury_amenities].isna().any(axis=1), luxury_amenities]
    df[df.sum(axis=1) == 0] = df[df.sum(axis=1) == 0].fillna(0)
    df[df.sum(axis=1) != 0] = df[df.sum(axis=1) != 0].fillna(
        df[df.sum(axis=1) != 0].median()
    )
    data.loc[df.index, luxury_amenities] = df

    data.loc[
        (data[luxury_amenities].sum(axis=1) > 0) & (data.CryoSleep.isna()), "CryoSleep"
    ] = False
    data.loc[
        (data[luxury_amenities].sum(axis=1) == 0) & (data.CryoSleep.isna()), "CryoSleep"
    ] = True

    data[["deck", "num", "side"]] = data.Cabin.str.split("/", expand=True)

    data.loc[(data.HomePlanet == "Earth") & (data.VIP.isna()), "VIP"] = False

    data[["group_num", "num_in_group"]] = data.PassengerId.str.split("_", expand=True)
    data.HomePlanet = data.groupby(["group_num"], sort=False)["HomePlanet"].apply(
        lambda x: x.ffill().bfill()
    )

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    cat_features = ["HomePlanet", "Destination", "deck", "side"]
    bin_features = ["CryoSleep", "VIP"]

    median_fill = ["Age"]
    most_frequent_fill = ["VIP", "HomePlanet", "Destination", "deck", "side"]

    data[median_fill] = pd.DataFrame(
        num_imputer.fit_transform(data[median_fill]), columns=median_fill
    )
    data[most_frequent_fill] = pd.DataFrame(
        cat_imputer.fit_transform(data[most_frequent_fill]), columns=most_frequent_fill
    )

    data[bin_features] = data[bin_features].astype(int)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    data = pd.concat(
        [
            data,
            pd.DataFrame(
                encoder.fit_transform(data[cat_features]),
                columns=encoder.get_feature_names_out(),
            ),
        ],
        axis=1,
    )

    data = data.drop(
        cat_features
        + [
            "PassengerId",
            "Cabin",
            "Name",
            "num",
            "group_num",
            "num_in_group",
            "side_S",
        ],
        axis=1,
    )

    data.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    make_dataset()
