from bs4 import BeautifulSoup
import requests
import pandas as pd

urlList = []
pageList = []
countList = []
companyNameList = []
tickerList = []
dateList = []
timeList = []
remarksList = []
qaList = []

z = 1
for page_num in range(1, 100):
    url = 'https://www.fool.com/earnings-call-transcripts/?page=' + str(page_num)
    response = requests.get(url)
    page = response.text
    soup = BeautifulSoup(page, "html.parser")

    for a in soup.find_all('a', {'class': 'flex-shrink-0 w-1/3 mr-16px sm:w-auto'}):
        urlList.append('https://www.fool.com' + str(a['href']))
        pageList.append(page_num)
        countList.append(z)
        z += 1
print("collected all websites")

for x in urlList:
    url = str(x)
    response = requests.get(url)
    page = response.text
    soup = BeautifulSoup(page, "html.parser")

    companyName = soup.find('div', {'class':"tailwind-article-body"}).find('strong').text.strip()
    companyNameList.append(companyName)

    ticker = soup.find('div', {'class':"tailwind-article-body"}).find('a', {'class': 'ticker-symbol'}).text.strip()
    tickerList.append(ticker)

    date = soup.find('div', {'class':"tailwind-article-body"}).find(id='date').text.strip()
    dateList.append(date)

    time = soup.find('div', {'class':"tailwind-article-body"}).find(id='time').text.strip()
    timeList.append(time)

    # Ignore the ad within prepared remarks
    ad_tags = soup.find_all('div', class_='article-pitch-container')
    for tag in ad_tags:
        tag.decompose()

    # Collect Prepared Remarks and
    body_tag = soup.find('div', {'class': "tailwind-article-body"})
    remarks_text = ''
    qa_text = ''
    skip_mode = True
    encountered_qa_header = False
    for tag in body_tag.contents:
        if tag.name == 'h2' and 'Prepared Remarks' in tag.text.strip():  # Skip all contents until Prepared Remarks header is found
            skip_mode = False  # Stop skipping once Header is found
            continue
        elif skip_mode:
            continue
        elif tag.name == 'h2' and 'Questions' in tag.text.strip() and 'Answer' in tag.text.strip():
            if not encountered_qa_header:  # Once Questions and Answer header is crossed begin storing text into qa_text
                encountered_qa_header = True
            continue
        elif tag.name:
            if encountered_qa_header:
                qa_text += tag.get_text(strip=True) + ' \n '
            else:
                remarks_text += tag.get_text(strip=True) + ' \n '
    remarksList.append(remarks_text)
    qaList.append(qa_text)
    print(f"Finished: {companyName}, {date}")


df = pd.DataFrame()
df['count'] = countList
df['page'] = pageList
df['companyName'] = companyNameList
df['ticker'] = tickerList
df['date'] = dateList
df['time'] = timeList
df['remarks'] = remarksList
df['qa'] = qaList
print(df)