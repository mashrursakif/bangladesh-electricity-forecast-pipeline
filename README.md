# Bangladesh Electricity Forecast Model Pipeline
Predict electricity generation and outage in Bangladesh using a machine learning model

## Pipeline
- Gather weather data using [open-meteo](https://open-meteo.com/) api
- Fetch past electricity data from [PGCB](https://erp.pgcb.gov.bd/w/generations/view_generations?page=1)
- Use aggregated weather and past electricity data as model inputs
- Forecast 7 days of electricity generation
- Forecast 7 days of loadshed and outages
- Saves daily updates and historical performance

Model outputs are in gigawatts (GW)

## Tools and Datasets Used
- LightGBM Regression Model for predicting power generation and loadshed
- GitHub Actions to perform daily updates
- Bangladesh electricity data from [PGCB](https://erp.pgcb.gov.bd/w/generations/view_generations?page=1)
- Weather Data from [High Volume Real-World Weather Data](https://data.mendeley.com/datasets/tbrhznpwg9/1) and [open-meteo](https://open-meteo.com/) api

## Author
Mashrur Sakif Souherdo - [GitHub](https://github.com/mashrursakif)
