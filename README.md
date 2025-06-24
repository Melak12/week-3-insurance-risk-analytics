# Week 3 - End-to-End Insurance Risk Analytics & Predictive Modeling
This week’s challenge centers on analyzing historical insurance claim data to help optimise the marketing strategy as well as discover “low-risk” targets for which the premium could be reduced. The analysis covers the following key areas:

- Insurance Terminologies
- A/B Hypothesis Testing
- Machine Learning & Statistical Modeling

### Business Objective
Your employer AlphaCare Insurance Solutions (ACIS) is committed to developing cutting-edge risk and predictive analytics in the area of car insurance planning and marketing in South Africa. You have recently joined the data analytics team as marketing analytics engineer, and your first project is to analyse historical insurance claim data. The objective of your analyses is to help optimise the marketing strategy as well as discover “low-risk” targets for which the premium could be reduced, hence an opportunity to attract new clients. 

The historical data is from Feb 2014 to Aug 2015.

### Keywords

1. **Premiums**: When you purchase an insurance policy, you'll be required to make regular payments, known as premiums. These payments are typically made monthly or annually and are the cost of maintaining your insurance coverage.
2. **Total Claim**: A claim is a formal request to your insurance company for coverage or reimbursement for a loss or damage. It's essential to follow the claims process outlined in your policy.
3. **No-Claims Discount**: Many insurance companies offer a no-claims discount to policyholders who haven't filed any claims within a specified period. This can lead to lower premiums over time.
4. **Deductible/Excess**: This is the amount of money you're responsible for paying out-of-pocket on a claim before your insurance coverage starts to pay. For example, if you have a $500 deductible and your car sustains $2,000 in damage, you'd pay the first $500, and your insurer would cover the remaining $1,500.
5. **Underwriting**: Underwriting is the process insurance companies use to assess the risk of insuring you or your property. It involves evaluating factors like your driving history, location, and the type of car you own to determine if the company will offer you coverage and at what premium.

### Insurance Terminologies

Check out the key insurance glossary [50 Common Insurance Terms and What They Mean — Cornerstone Insurance Brokers](https://www.cornerstoneinsurancebrokers.com/blog/common-insurance-terms)

## Setup
1. Clone: `git clone https://github.com/Melak12/week-3-insurance-risk-analytics.git`
2. Create venv: `python3 -m venv .venv`
3. Activate: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`

Note: if requirements.txt is missing, you might need to run this command
`pip freeze > requirements.txt`

---

## Data Version Control (DVC) Usage

This project uses [DVC (Data Version Control)](https://dvc.org/) to manage large datasets and track data pipeline stages efficiently.

### DVC Installation
Install DVC using pip:
```sh
pip install dvc
```

### Initialize DVC in Your Project
If not already initialized, run:
```sh
dvc init
git commit -m "Initialize DVC"
```

### Adding Data to DVC
To track your data files (e.g., in the `data/` directory):
```sh
dvc add data/MachineLearningRating_v3.txt
git add data/MachineLearningRating_v3.txt.dvc .gitignore
git commit -m "Track raw insurance data with DVC"
```

### Sharing Data
To push your data to remote storage (e.g., Google Drive, S3, Azure):
```sh
dvc remote add -d myremote <remote-storage-url>
dvc push
```

### Pulling Data
To retrieve data tracked by DVC (after cloning the repo):
```sh
dvc pull
```

### Reproducing Pipelines
If you use DVC pipelines (e.g., for data processing or modeling):
```sh
dvc repro
```

### More Information
- [DVC Documentation](https://dvc.org/doc)
- [Get Started with DVC](https://dvc.org/doc/start)

---