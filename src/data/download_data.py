import time
import numpy as np
import pandas as pd
import os

# Function to filter the raw data 
def isolate_city_state(data, cities, states):
    """ This ensures that each city is selected with it's respective state
    which is why I didn't simply run a merge statement. """

    for ind, x in enumerate(zip(cities, states)):
        tmp = data.loc[(data['City'] == x[0]) & (data['State'] == x[1])].copy()
        if ind == 0:
            new_data = tmp
        else:
            new_data = new_data.append(tmp)
    return new_data

# Download the raw data and filter it according to the cities
def download_filter_data(url, city_list, state_list):
    start_time = time.time()
    print("downloading raw file ~550 mb ... ")
    accident_df = pd.read_csv(url)
    download_time = time.time() - start_time
    print(f"Data Download Time {download_time} secs")
    six_cities_df = isolate_city_state(accident_df, city_list, state_list)
    return six_cities_df


def generate_base_data():
    URL = 'https://www.dropbox.com/s/mdw2asjrh8bm038/US_Accidents_Dec20_updated.csv?dl=1'
    city_list = ['Phoenix', 'Los Angeles', 'New York', 'Philadelphia', 'Houston', 'Chicago']
    state_list = ['AZ', 'CA', 'NY', 'PA', 'TX', 'IL']

    # Construct the relative file path
    dirname = os.path.dirname(__file__)
    relative_path = '../../data/raw/accident_data.csv'
    datafile = os.path.join(dirname, relative_path)

    # Delete if the file with the same name exists
    if os.path.exists(datafile):
        os.remove(datafile)
        print("old csv file deleted")
    else:
        print("no old csv file exists")

    six_cities_df = download_filter_data(URL, city_list, state_list)
    six_cities_df.to_csv(datafile, index=False)
    
    print("Job completed, check the filtered csv file in data/processed/ folder.")