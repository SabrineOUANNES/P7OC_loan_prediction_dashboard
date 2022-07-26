import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def get_boxplot_comparison(melt_data, client_data, features):
    fig, ax = plt.subplots(figsize=(30, 10))
    ax = sns.boxplot(data=melt_data,
                     x="variable",
                     y="value",
                     hue='TARGET',
                     palette=['red', 'green'],
                     )
    plt.xlabel('Variable', size=30)
    plt.ylabel('Values', size=30)
    plt.xticks(rotation=75, size=25)
    plt.yticks(size=25)
    plt.ylim(-5, 5)
    ax = sns.swarmplot(data=client_data[features], color='b',
                       marker='o', size=20, edgecolor='k', label='applicant customer', ax=ax)
    # legend
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h[:3], fontsize=20)
    st.pyplot(fig)


def get_hist_comparison(features, data, client_data):
    fig = plt.figure(figsize=(20, 50))
    for i, label in enumerate(features):
        plt.subplot(11, 2, i + 1)
        sns.histplot(data=data,
                     x=data[label],
                     edgecolor='none',
                     bins=30,
                     hue='TARGET',
                     palette=['red', 'green'],
                     legend=False)
        plt.legend(title='Loan prediction', loc='upper left', labels=['REPAID', 'NOT REPAID'])
        plt.yticks(size=7)
        plt.ylabel('Count', size=15)
        plt.xticks(size=7)
        plt.xlabel(label, size=15)
        plt.axvline(x=client_data[label].values, ymin=data[label].min(), ymax=data[label].max(), color='b')
    st.pyplot(fig)

