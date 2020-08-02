# Starbucks-Capstone
Project in Data Scientist Nanodegree of Udacity

### Table of Contents

1 . [Installation](#installation)

2 . [Project Motivation](#motivation)

3 . [File Descriptions](#files)

4 - [Exploratory Analysis](#Analysis)

5 - [Recommendation Engine](#Recommendation)

6 . [Blog Website](#Blog)


## Installation <a name="installation"></a>



There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

Visit [localhost:8888](http://localhost:8888) in a web browser to
access Jupyter Notebooks.

## Project Motivation<a name="motivation"></a>

It is the Starbuck's Capstone Challenge of the Data Scientist Nanodegree in Udacity. We get the dataset from the program that creates the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers. We want to make a recommendation engine that recommends Starbucks which offer should be sent to a particular customer.

We are interested to answer the following two questions:
1 - What is the population distribution by gender?
2 - Which are the most popular offer.?
3 - Which gender group has done the most expensive transaction.?
4 - What is the relation between income and transaction.?
5 - The relation between Age and Total expenses.?


## File Descriptions <a name="files"></a>

The notebook available here showcases work related to the above questions.  

This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

The data is contained in three files:
- `portfolio.json` - containing offer ids and meta data about each offer (duration, type, etc.)
- `profile.json` - demographic data for each customer
- `transcript.json` - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

`portfolio.json`
- id (string) - offer id
- offer_type (string) - the type of offer ie BOGO, discount, informational
- difficulty (int) - the minimum required to spend to complete an offer
- reward (int) - the reward is given for completing an offer
- duration (int) - time for the offer to be open, in days
- channels (list of strings)

`profile.json`
- age (int) - age of the customer
- became_member_on (int) - the date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

`transcript.json`
- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
- person (str) - customer id
- time (int) - time in hours since the start of the test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record


## Exploratory Analysis <a name="Analysis"></a>

In this stage, we analyzed the population based on their demographics and their spending behavior. We also took into account the interactions between the customers and the offers provided.

## Recommendation Engine <a name="Recommendation"></a>
We used a knowledge based recommendation engine in this project. We provided one that selects the most popular offers without considering demographics, first. This system is a good start for customers that do not provide any demographic data in the app.

For the rest of costumers, we introduced filters that help the system make recommendations based on the demographic data provided by the customers.

We evaluated the recommendation systems by using visualizations from the data.
## Blog Website <a name="Blog"></a>
A blog has been published  [Analyzing The Starbuck Offers](https://medium.com/@ashwanisng/analyzing-the-starbuck-offers-9ab7653456a8)
