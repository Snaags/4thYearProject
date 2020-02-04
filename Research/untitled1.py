# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:29:56 2019

@author: chris
"""

from selenium import webdriver
import string
import bs4

browser = webdriver.Chrome("C:\\Users\\chris\\Downloads\\chromedriver_win32\\chromedriver")

browser.get("https://www.advfn.com/nyse/newyorkstockexchange.asp?companies=A")

TAGS = []
tags = []
datahold = 1
#print(Tags)
for i in string.ascii_uppercase:
    browser.get("https://www.advfn.com/nyse/newyorkstockexchange.asp?companies="+i)
    
    html = browser.page_source
    soup = bs4.BeautifulSoup(html)
    
    x = soup.body.table.next_sibling.next_sibling.next_sibling.next_sibling.tbody \
        .tr.next_sibling.next_sibling.td.next_sibling.next_sibling.table.tbody.tr \
        .td.table.tbody.tr.next_sibling.next_sibling.next_sibling.next_sibling
    
    while x != None:
        datahold = x.td.next_sibling.a
        tags.append(datahold)
        x = x.next_sibling.next_sibling
   
for i in tags:
    if 'stock-price">' in str(i):
        TAGS.append(str(i)[str(i).find('stock-price">')+13:].strip("</a>"))
    
    
    
    

    
#[contains(@rel, "next")]

#elem = browser.find_element_by_class_name("pagination")

"/html/body/table[2]/tbody/tr[2]/td[2]/table/tbody/tr/td/table/tbody"



#elem.click()
