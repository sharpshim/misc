#!/usr/bin/python
#-*- coding: utf-8 -*-

import urllib2
import re
import time
import sys

SLEEP_TIME = 1
PAGE_ENCODING = "euc-kr"
NEWS_SITE = "nate"
news_url_pattern = re.compile(r"http://news.nate.com/view/20[0-9]{6}n[0-9]{5}\?mid=n[0-9]{4}")
list_url_pattern = re.compile(r"/subsection[^\"]*page=[0-9]+")
category_pattern = re.compile(r"mid=(n[0-9]{4})")

def crawl_page(url) :
    f = urllib2.urlopen(url)
    url_body = f.read()
    time.sleep(SLEEP_TIME)
    return unicode(url_body, PAGE_ENCODING).encode("utf-8")

def strip_html(line):
    line = line.replace("<br />", " ")
    line  = line.replace("\r\n", " ")
    line  = line.replace("\n", " ")
    p = re.compile(r'<.*?>')
    line = p.sub("", line)
    p = re.compile(r" ( )*")
    line  = p.sub(" ", line)
    return line.strip()

class page :
    def __init__(self, url, page_type, url_body) :
        print >> sys.stderr, "processing %s %s" % (url, page_type)
        self.url = url
        self.page_type = page_type
        if page_type == "news" :
            self.news_body = self.get_news_body(url_body)
            self.news_title = self.get_news_title(url_body)
        elif page_type == "list" :
            news_urls, list_urls = self.get_urls(url_body)
            # for nate - catcode filtering
            cat_code = category_pattern.search(url).group(1)
            news_urls = [(x, "news") for x in news_urls if category_pattern.search(x).group(1) == cat_code]
            list_urls = [("http://news.nate.com" + x.replace("&amp;", "&"), "list") for x in list_urls]

            self.urls = news_urls + list_urls

    def get_news_body(self, url_body) :
        # for nate news
        if NEWS_SITE == "nate" :
            idx1 = url_body.find("google_ad_section_start(name=gad1)") + 41
            idx2 = url_body.find("google_ad_section_end(name=gad1)", idx1) - 9
            news_body = url_body[idx1:idx2]
        return news_body.replace("\t", " ") 

    def get_news_title(self, url_body) :
        # for nate news
        if NEWS_SITE == "nate" :
            idx1 = url_body.find("<h3 class=\"articleSubecjt\">") + 27
            idx2 = url_body.find("</h3>", idx1+1)
            news_title = url_body[idx1:idx2]
        return news_title

    def get_urls(self, url_body) :
        list_urls = list_url_pattern.findall(url_body)
        news_urls = news_url_pattern.findall(url_body)
        
        return news_urls, list_urls

url_set = set([])

def initial_urls() :
    the_list = []
    for line in file("/home/sharpshim/projects/news_crawl/initial_urls.txt") :
        if line.strip().startswith("#") or line.strip() == "":
            continue
        the_list.append((line.strip(), "list"))
    return the_list

if __name__ == "__main__" :
    fop = open(sys.argv[1], "w")
    url_queue = initial_urls()
    while len(url_queue) > 0 :
        url, page_type = url_queue.pop(0)
        try :
            new_page = page(url, page_type, crawl_page(url))
        except UnicodeDecodeError :
            print >> sys.stderr, "Encoding Exception : %s" % url
            continue

        if page_type == "news" :
            fop.write("%s\t%s\t%s\n" % (url, new_page.news_title.strip(), strip_html(new_page.news_body.strip())))
        elif page_type == "list" :
            for u in new_page.urls :
                if u[0] not in url_set :
                    print >> sys.stderr, "add queue", u
                    url_queue.append(u)
                    url_set.add(u[0])
                else :
                    print >> sys.stderr, "%s is in the set" % u[0]
    fop.close()

