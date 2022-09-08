import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import textwrap

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class Runner:

    plot_face_color = "lightsteelblue"

    date_columns = [
        "accident_date",
        "ancr_date",
        "assembly_date",
        "c-2_date",
        "c-3_date",
        "controverted_date",
        "first_appeal_date",
        "first_hearing_date",
        "ppd_non-scheduled_loss_date",
        "ppd_scheduled_loss_date",
        "ptd_date",
        "section_32_date",
        ]

    yn_columns = ["accident", "alternative_dispute_resolution", "attorney/representative", "occupational_disease", "covid-19_indicator"]
    
    edge_c = matplotlib.colors.colorConverter.to_rgba('black', alpha=.3)

    def __init__(self, df) -> None:
        self.df = df
        self.zip_df = None
        self.plots = {}
        self.analysis_data = {}

    def clean_df(self):
        
        self.df.columns = [c.replace(" ", "_") for c in self.df.columns.str.lower().tolist()]
        for column in self.date_columns:
            self.df[column] = pd.to_datetime(self.df[column])

        for column in self.yn_columns:
            self.df[column] = np.where(self.df[column]=="Y", 1, 0)
        

        self.df["accident_to_assembly"] = (self.df["assembly_date"] - self.df["accident_date"]).dt.days
        self.df["gender"] = np.where(self.df["gender"] == "M", 1, 0)
        self.df["ime-4_count"] = np.where(self.df["ime-4_count"].isna(), 0, self.df["ime-4_count"])

        logic = (self.df.accident_date >= '2000-01-01') & (self.df.accident_to_assembly >= 0) & (self.df["accident_to_assembly"] < 500)

        self.df = self.df[logic]

    def map_analysis(self, zip_map):
            counts = self.df.groupby("zip_code").count()["claim_identifier"]
            counts_df = pd.DataFrame({"count": counts}).reset_index()
            merged = zip_map.merge(counts_df, left_on="ZCTA5CE10", right_on="zip_code")
            self.zip_df = merged

    def census_analysis(self, census_df):
        self.census_df = census_df
        new_idx = [str(i)[6:11] for i in self.census_df.index]
        census_series = pd.Series({i: d for i, d in zip(new_idx, self.census_df["B01003_001E"])})
        self.zip_df = self.zip_df.merge(pd.DataFrame({"population": census_series}).reset_index(), left_on="ZCTA5CE10", right_on="index")        
        self.zip_df["population"] = np.where(self.zip_df["population"] < 1000, 0, self.zip_df["population"])
        self.zip_df["claims_per_capita"] = (self.zip_df["count"] / self.zip_df["population"] * 1000).replace([np.inf, -np.inf], 0)


    def map_plot(self, map, title, xlim=(-74.3, -73.5), ylim=(40.45,41), is_zip_map=True, target_col="count"):
        fig, axs = plt.subplots(ncols=2, figsize = (30, 10), facecolor=self.plot_face_color)
        if is_zip_map:
            fig.suptitle(title, fontsize=40)
            axs[1].set_xlim(xlim)
            axs[1].set_ylim(ylim)
            axs[1].xaxis.set_visible(False)
            axs[1].yaxis.set_visible(False)
            axs[0].xaxis.set_visible(False)
            axs[0].yaxis.set_visible(False)
            axs[0].set_facecolor("lightblue")
            axs[0].set_title("Statewide", fontdict={"fontsize": 20})
            axs[1].set_facecolor("lightblue")
            axs[1].set_title("Metropolitan Area", fontdict={"fontsize": 20})
            self.zip_df.plot(ax=axs[0], column=target_col, cmap="Reds", edgecolor=self.edge_c, legend="True", missing_kwds= dict(color = "lightgrey",))
            self.zip_df.plot(ax=axs[1], column=target_col, cmap="Reds", edgecolor=self.edge_c, legend="True", missing_kwds= dict(color = "lightgrey",))
            plot_name = f"zip_map-{target_col}"
            fig.savefig(f"plots/{plot_name}.png", dpi=300)
            self.plots[plot_name] = fig
        else:
            county_counts = self.df.groupby("county_of_injury").count()["claim_identifier"]
            county_counts_df = pd.DataFrame({"count": county_counts}).reset_index()
            county_merge = map.merge(county_counts_df, left_on="name", right_on="county_of_injury")
            fig, ax = plt.subplots(figsize = (10, 10), facecolor=self.plot_face_color)
            ax.set_facecolor("lightblue")
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            fig.suptitle(title)
            county_merge.plot(ax=ax, column="count", cmap="Reds", edgecolor=self.edge_c, legend="True", missing_kwds= dict(color = "lightgrey",))
            fig.savefig("plots/county_map.png", dpi=300)
            self.plots["county_map"] = fig

    def _calc_ranks(self):
        reasons = [
            "wcio_nature_of_injury_description",
            "oiics_part_of_body_description",
            "wcio_cause_of_injury_description",
            "oiics_nature_of_injury_description",
            "wcio_part_of_body_description",
            "oiics_injury_source_description",
        ]
        self.descriptions = {reason[:-12]: self.df.groupby(reason).count()["claim_identifier"].sort_values(ascending=False) for reason in reasons}
        
    def bar_plot(self, top_n=10):
        fig, axs = plt.subplots(figsize=(80, 30), nrows=2, ncols=3, facecolor=self.plot_face_color)
        fig.suptitle("WCIO and OIICS Injury Classifications", fontsize=80)
        fig.subplots_adjust(bottom=0.2, hspace=0.7)
        i = 0
        for k in list(self.descriptions.keys()):
            r = i % 2
            c = i % 3
            target = self.descriptions[k]
            labels = [textwrap.fill(c, 20) for c in target.index]
            title = k.replace("_", " ").upper()
            ax = axs[r][c]
            ax.set_xticklabels(labels[0:top_n-1], fontdict={"fontsize": 20}, rotation=45)
            ax.set_title(title, fontdict={"fontsize": 30})
            ax.bar(target.index[0:(top_n-1)], target[0:(top_n-1)])
            ax.set_ylabel("Number of Claims", fontdict={"fontsize": 20})
            i += 1
        fig.savefig(f"plots/bar.png")
        self.plots["bar"] = fig

    def calculate_ts(self):
        self.df["accident_date_month_trunc"] = self.df["accident_date"] + pd.offsets.MonthBegin(-1)
        self.by_day = self.df.groupby("accident_date").count()["claim_identifier"]
        self.by_month = self.df.groupby("accident_date_month_trunc").count()["claim_identifier"]


    def time_series_plot(self, xlim=("2000-01-01", "2022-05-01"), outname="overall", just_daily=False):
        if just_daily:
            fig, ax = plt.subplots(figsize=(25, 10), facecolor=self.plot_face_color)
        else:
            fig, axs = plt.subplots(ncols=2, figsize=(25, 10), facecolor=self.plot_face_color)


        month_logic = (self.by_month.index >= xlim[0]) & (self.by_month.index <= xlim[1])
        day_logic = (self.by_day.index >= xlim[0]) & (self.by_day.index <= xlim[1])

        if just_daily:
            ax.set_title("Number of Claims by Day", fontdict={'fontsize': 20})
            ax.grid(color=self.edge_c, alpha=0.2)
            ax.plot(self.by_day[day_logic])
        else:
            axs[0].set_title("Number of Claims by Month", fontdict={'fontsize': 20})
            axs[0].grid(color=self.edge_c, alpha=0.2)
            axs[0].plot(self.by_month[month_logic])

            axs[1].set_title("Number of Claims by Day", fontdict={'fontsize': 20})
            axs[1].grid(color=self.edge_c, alpha=0.2)
            axs[1].plot(self.by_day[day_logic])

        fig.savefig(f"plots/{outname}.png")
        self.plots[f"ts-{outname}"] = fig

    def attorney_analysis(self):
        pivoted = self.df.pivot_table(values="claim_identifier", index="highest_process", columns="attorney/representative", aggfunc=len)
        pivoted_pct = pivoted / pivoted.sum()
        self.analysis_data["attorney_pct"] = pivoted_pct
        model_df = self.df


        logic = (self.df.average_weekly_wage >= 50) & (self.df.average_weekly_wage <= 5000) & (self.df.age_at_injury >= 18) & (self.df.age_at_injury <= 65)
        model_df = self.df[logic]
        y = np.where(model_df["highest_process"] == "4A. HEARING - JUDGE", 1, 0)
        X = model_df[["attorney/representative"]]
        simple_regression = LogisticRegression()
        simple_regression.fit(X, y)
        self.analysis_data["simple_regression"] = simple_regression

        y = np.where(model_df["highest_process"] == "4A. HEARING - JUDGE", 1, 0)
        X = model_df[["attorney/representative", "age_at_injury", "average_weekly_wage", "gender", "ime-4_count"]]

        scaler = StandardScaler()
        logistic_regression = LogisticRegression()

        X_scale = scaler.fit_transform(X)
        logistic_regression.fit(X_scale, y)
        self.analysis_data["regression"] = logistic_regression

    def plot_density(self, column="accident_to_assembly"):
        fig, ax = plt.subplots(facecolor=self.plot_face_color)
        fig.suptitle(f"Histogram: {column}")
        ax.hist(self.df[column], bins=300, edgecolor=self.edge_c)
        ax.set_xlim(right=300)
        ax.set_ylabel(column)
        self.plots[f"density-{column}"] = fig

    def final_plot(self):
        median_groupby = self.df.groupby("assembly_date").agg(np.median)["accident_to_assembly"]
        mean_groupby = self.df.groupby("assembly_date").agg(np.mean)["accident_to_assembly"]

        fig, axs = plt.subplots(ncols=2, figsize=(25, 10), facecolor=self.plot_face_color)

        axs[0].set_title("Mean Time Between Assembly Date and Accident Date (Days)", fontdict={'fontsize': 20})
        axs[0].set_xlabel("Assembly Date")
        axs[0].set_ylabel("Mean Claim Days Since Accident")
        axs[0].grid(color=self.edge_c, alpha=0.2)
        axs[0].plot(mean_groupby["2019-07-01":])

        axs[1].set_title("Median Time Between Assembly Date and Accident Date (Days)", fontdict={'fontsize': 20})
        axs[1].set_xlabel("Assembly Date")
        axs[1].set_ylabel("Median Claim Days Since Accident")
        axs[1].grid(color=self.edge_c, alpha=0.2)
        axs[1].plot(median_groupby["2019-07-01":])

        self.plots[f"final"] = fig
        fig.savefig(f"plots/final.png")

    def pickle_plots(self):
        pickle.dump(self.plots, open("plots/plots.pickle", "wb"))

    def pickle_data(self):
        pickle.dump(self.analysis_data, open("data/analysis.pickle", "wb"))