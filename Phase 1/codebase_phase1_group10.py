import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import time
import


######################################################################################################
###################################   USER INPUTS   ##################################################
######################################################################################################

# Specifies the topic that is assigned to us on which news articles are to be scraped out from
keyword = "climate change"

# CNN news website for scraping out the news content
cnn_url = "https://www.cnn.com/search?q=" + keyword
nytimes_url = "https://www.nytimes.com/search?query=" + keyword

# User input
article_countdown = 100
num_article = article_countdown

# Limit for the machine generated text word count for the GPT-2 model (top-k samples and max_count)
machine_text_word_count = 300
numberof_top_k_samples = 300

# Choice to display the human generated text output or not
DISPLAY_OUTPUT = False
# Choice to display the machine generated text output or not
DISPLAY_MACHINE_TEXT = True
# Choice to generate the GPT-2 based machine generated text output or not
GPT2 = True
# Choice to generate the Deep AI based machine generated text output or not
DEEP_AI = False
# Give True for CNN scraping and False for NYTIMES scraping
url_bool = False
# Choice to display the machine generated text output or not for Nytimes
DISPLAY_NYTIMES = True

######################################################################################################
###################################   VARIABLE DECLARATIONS ##########################################
######################################################################################################

# Program requirements
NEXT_PAGE = '.pagination-arrow.pagination-arrow-right.cnnSearchPageLink.text-active'
new_page_not_loaded = True
load_count = 1

# Output variables
headings_list = []
content_list = []
machine_text_list = []
links = []
headings_list_temp = []
article_countdown_temp = article_countdown

# This value was obtained by testing and manual checking, but doesn't change under any circumstance during the execution and ensured to be a constant
UNWANTED_NYTIMES_LINKS = 150

######################################################################################################
###################################   SELENIUM WEBDRIVER SETUP - Raakesh #############################
######################################################################################################

if(url_bool):
    url = cnn_url
else:
    url = nytimes_url

chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
wd = webdriver.Chrome('chromedriver', options=chrome_options)
wd.get(url)
updated_wd = wd

######################################################################################################
#####################   CNN NEWS SCRAPING - Raakesh ###########################
######################################################################################################

html_content = updated_wd.page_source
soup = BeautifulSoup(html_content, 'html5lib')

if(url_bool):
    # news_headings = soup1.find_all('a')
    news_headings = soup.find_all('h3', class_="cnn-search__result-headline")
    news_content = soup.find_all('div', class_="cnn-search__result-body")
    # print(news_headings)

    # Collecting the required number of articles
    while (article_countdown > 0):
        article_countdown -= len(news_headings)
        first_heading = news_headings[0].get_text().strip()

        # Collecting the news article for each and every heading
        for index, heading in enumerate(news_headings):
            heading_text = heading.get_text()
            heading_text = heading_text.strip()
            headings_list.append(heading_text)
            content_list.append(news_content[index].get_text().strip())

        # Click on the "Next" button for next 10 articles
        wd.find_element(By.CSS_SELECTOR, ".pagination-arrow.pagination-arrow-right.cnnSearchPageLink.text-active").click()

        # load_count denotes the number of articles per page and the loop doesn't end until all the articles within the page has been extracted
        while (new_page_not_loaded and load_count != 10):
            html_content = wd.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            # news_headings = soup1.find_all('a')
            news_headings = soup.find_all('h3', class_="cnn-search__result-headline")
            news_content = soup.find_all('div', class_="cnn-search__result-body")

            # Check if the next page has been loaded
            if (news_headings[0].get_text() == first_heading):
                new_page_not_loaded = True
            else:
                new_page_not_loaded = False
            load_count += 1

######################################################################################################
##################  NYTIMES NEWS SCRAPING - Raakesh ###########################
######################################################################################################

else:
    li_items = soup.find_all('li')

    # Here 1.5 is multiplied with the countdown as half of the articles were found to be videos from which text content could not be extracted
    while (len(li_items) - UNWANTED_NYTIMES_LINKS < int(1.5 * article_countdown)):
        wd.find_element(By.XPATH, '//button[text() = "Show More"]').click()
        updated_wd = wd
        html_content = updated_wd.page_source
        soup = BeautifulSoup(html_content, 'html5lib')
        li_items = soup.find_all('li')

    # Extract the links to the pages corresponding to each and every heading
    for ind, li in enumerate(li_items):
        news_heading_withtag = li.find_all('h4')
        if (len(news_heading_withtag) > 0):
            headings_list_temp.append(news_heading_withtag[0].get_text())
            links.append("https://www.nytimes.com/" + li.find('a')['href'])

    headings_list_temp = headings_list_temp[1:]
    links = links[1:]

    # Iterate through the extracted links to get the news content
    for index, url in enumerate(links):

        # Print for intuition
        print("Extracting article: " + str(index + 1))

        wd.get(url)
        updated_wd = wd

        html_content = updated_wd.page_source
        soup2 = BeautifulSoup(html_content, 'html5lib')

        news_content = soup2.find('section', itemprop="articleBody")

        # Condition to check as some news were just video content
        if (news_content is not None):

            extracted_text = news_content.get_text()

            # Filter out the times signature on the webpage template for NYTimes
            re_obj = re.search("The Times", extracted_text)
            if (re_obj is not None):
                char_index = re_obj.span(0)[0]
                extracted_text = extracted_text[:char_index]

            # Store the content and exist loop if countdown exceeds required count
            content_list.append(extracted_text)
            headings_list.append(headings_list_temp[index])
            article_countdown_temp -= 1
            if (article_countdown_temp == 0):
                break

    # Conditional display of the news headings and contents at output
    if (DISPLAY_NYTIMES):
        for index, item in enumerate(headings_list):
            print("\n\nHeading: ")
            print(item)
            print("\nContent: ")
            print(content_list[index])

# Display the final lists
if(DISPLAY_OUTPUT):
    for i in range(len(headings_list)):
        print("\n")
        print("News heading: ")
        print(headings_list[i] + " : ")
        print("News content: ")
        print(content_list[i])

######################################################################################################
###################################   GPT2 MODEL FOR MACHINE TEXT - Raakesh  #########################
######################################################################################################

if(GPT2):

    print("Machine generated text (gpt-2):\n\n ")

    # Init the tokenizer object that tokenizes the text content from the previously trained GPT-2 175M model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # add the EOS token as PAD token to avoid warnings
    model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

    # Create the table for storing the machine generated values
    table = pd.DataFrame(columns=('Headline', 'Human_text', 'Human_text_source', 'Machine_text', 'Machine_text_source'))

    try:
        for i, heading in enumerate(headings_list):

            print("Generating article (GPT-2) " + str(i))

            # encode context the generation is conditioned on ; Here the context is the heading of the news article
            input_ids = tokenizer.encode(heading, return_tensors='tf')

            # set seed to reproduce results
            tf.random.set_seed(0)

            # set top_k for top #samples of output text and set top_p for the top results above the given CDF [0, 1]. 0.95 was found to be the optimum value according to the papers
            sample_outputs = model.generate(
                input_ids,
                do_sample=True,
                max_length=machine_text_word_count,
                top_k=numberof_top_k_samples,
                top_p=0.95,
                num_return_sequences=1
            )

            print("Output:\n" + 100 * '-')

            # Apply the decoder to run the tests
            decoded_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True).replace('\n', '')
            # Use regex to format the text to remove the last few words that consitute an incomplete sentence
            complete_text = re.findall(".*\.", decoded_text)[0]
            # Append the generated text to a list
            machine_text_list.append(complete_text)

            # input the headings into a series to be entered into a dataframe
            series = pd.Series({'Headline': headings_list[i], 'Human_text': content_list[i], 'Human_text_source': 'nytimes', 'Machine_text': complete_text, 'Machine_text_source': 'GPT-2'})
            table = table.append(series, ignore_index=True)

            # Display the output of the machine generated text
            if(DISPLAY_MACHINE_TEXT):
                print("Machine generated text for article " + str(i) + ": ")
                print("News heading: ")
                print(headings_list[i])
                print("News content: ")
                print(machine_text_list[i] + "\n\n")

    except:
        print("An indexing error occured...")

    finally:
        # Write the output to a csv
        data_df = pd.DataFrame(table)
        data_df.to_csv('cse472_database_gpt2_nytimes.csv', index=False, header=True, encoding="utf-8-sig")
        print("Data successfully written to cse472_database_gpt2_nytimes.csv file...")

######################################################################################################
###################################   DEEP AI API FOR MACHINE TEXT  - Dunchuan #######################
######################################################################################################

if (DEEP_AI):

    print(100*"-" + "\n\n")
    print("Deep AI generated text...\n\n")

    # pytorch API for generating machine text
    fake_text_list = []  # store machine text in list
    for i in range(num_article):  # [0, num_article-1]
        t0 = time.process_time()
        # print("Current articel: " + str(i))
        print("\n")
        print("News heading: " + headings_list[i])
        r = requests.post(
            "https://api.deepai.org/api/text-generator",
            data={
                'text': headings_list[i],
            },
            headers={'api-key': '8497ebd0-687c-4065-a155-3b10e78df7f1'}
        )
        print("News content: ")
        print(r.json()['output'])
        t1 = time.process_time() - t0
        print("Time elapsed for iteration: ", t1)
        fake_text_list.append(r.json())

        table = pd.DataFrame(
            columns=('Headline', 'Human_text', 'Human_text_source', 'Machine_text', 'Machine_text_source'))
        for i in range(num_article):
            str = pd.Series({'Headline': headings_list[i], 'Human_text': content_list[i], 'Human_text_source': 'cnn',
                             'Machine_text': fake_text_list[i], 'Machine_text_source': 'DeepAI'})
            table = table.append(str, ignore_index=True)
        # print(table)

        data_df = pd.DataFrame(table)
        data_df.to_csv('cse472_database.csv', index=False, header=True, encoding="utf-8-sig")