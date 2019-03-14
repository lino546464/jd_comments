
import requests
import random
import time
import json
import re
import csv
import codecs


#base_url = 'https://rate.tmall.com/list_detail_rate.htm?itemId=521270814585&spuId=347093238&' \
#      'sellerId=720062993&order=3&currentPage='

#base_url = 'https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv14056&productId=4000128&score=3&sortType=5&pageSize=10&page='

lis = ['787623','1639078','787656','787632','787638','1431709','787628','3603359']
scores = ['1','3']

#proxies = {'http':'125.126.202.159'}

#cookies = '_uab_collina=155227047228464738739746; thw=cn; cna=eLIMFfBkmwMCAbSoupbiEIet; t=c3f819315eeff6fbc7dcdfb979932879; lc=VypS3zMd9srIf3Iho7wAAPvz2A%3D%3D; tg=0; enc=5xOeeClhGr9ovvAJGMuBKH1paXd0MFWJ3R0lM%2FiIcuwqyqiT3%2BIKBMmIJePYo78kLV%2BNx4ERK2ovyOYuQmhkBg%3D%3D; hng=CN%7Czh-CN%7CCNY%7C156; XSRF-TOKEN=be420737-025d-4e3c-bb2c-89e7beb303ae; cookie2=196cd3a723790da2199346b5c74e5484; _tb_token_=e5fe33a33336; _cc_=UIHiLt3xSw%3D%3D; x=e%3D1%26p%3D*%26s%3D0%26c%3D0%26f%3D0%26g%3D0%26t%3D0%26__ll%3D-1%26_ato%3D0; l=As3NGxTNqp8P4M7OFmulEkOrXeJHqgF8; whl=-1%260%260%261552285908450; mt=ci=0_0; cookieCheck=49159; v=0; isg=BAMDf5QNW-zAHhcYJQpjZZMNksdt0Je3SkKyhzXg52L99CYWrUmNC7KmbsQf1O-y'

user_agent = [
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
    "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (Linux; U; Android 2.3.7; en-us; Nexus One Build/FRF91) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
    "MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
    "Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10",
    "Mozilla/5.0 (Linux; U; Android 3.0; en-us; Xoom Build/HRI39) AppleWebKit/534.13 (KHTML, like Gecko) Version/4.0 Safari/534.13",
    "Mozilla/5.0 (BlackBerry; U; BlackBerry 9800; en) AppleWebKit/534.1+ (KHTML, like Gecko) Version/6.0.0.337 Mobile Safari/534.1+",
    "Mozilla/5.0 (hp-tablet; Linux; hpwOS/3.0.0; U; en-US) AppleWebKit/534.6 (KHTML, like Gecko) wOSBrowser/233.70 Safari/534.6 TouchPad/1.0",
    "Mozilla/5.0 (SymbianOS/9.4; Series60/5.0 NokiaN97-1/20.0.019; Profile/MIDP-2.1 Configuration/CLDC-1.1) AppleWebKit/525 (KHTML, like Gecko) BrowserNG/7.1.18124",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Trident/5.0; IEMobile/9.0; HTC; Titan)",
    "UCWEB7.0.2.37/28/999",
    "NOKIA5700/ UCWEB7.0.2.37/28/999",
    "Openwave/ UCWEB7.0.2.37/28/999",
    "Mozilla/4.0 (compatible; MSIE 6.0; ) Opera/UCWEB7.0.2.37/28/999",
    # iPhone 6：
	"Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25"]

def get_user_agent():
    return random.choice(user_agent)

'''
def w_cookie(cookies):
    cookiee = {}
    for cookie in cookies.split(';'):
        name, value = cookie.strip().split('=', 1)
        cookiee[name] = value
    return cookiee
'''

def write_csv(file_name, datas):
    f = codecs.open(file_name,'a','utf-8')
    writer = csv.writer(f)
    writer.writerows(datas)

def get_page(headers,base_url):
    first_url = base_url + '0'
    r = requests.get(url=first_url, headers=headers)
    # print(r.text)
    pettrn = re.compile('\((.*)\)', re.S)
    con = re.findall(pettrn, str(r.text))

    jc = json.loads(str(con[0]))
    # print(jc.keys())
    contents = []
    # content = {}
    #details = jc.get('rateDetail').get('rateList')
    page = jc.get('maxPage')
    print('共有{}页评论'.format(page))
    return int(page)

def get_comments(url,headers,path):
    r = requests.get(url=url, headers=headers)
    #print(r.text)
    pettrn = re.compile('\((.*)\)', re.S)
    con = re.findall(pettrn, str(r.text))

    jc = json.loads(str(con[0]))
    #print(jc.keys())
    contents = []
    #content = {}
    details = jc.get('comments')
    #page = jc.get('rateDetail').get('paginator').get('lastPage')
    for detail in details:
        id = detail.get('id')
        guid = detail.get('guid')
        time = detail.get('creationTime')
        content = detail.get('content')
        s = [id,guid, time, content]
        contents.append(s)
    print(contents)
    write_csv(path,contents)
    #time.sleep(5)


if __name__ == '__main__':
    #cookiee = w_cookie(cookies)
    lose = []
    for li in lis:
        print("正在爬取商品id:{}".format(li))
        for score in scores:
            if score == "1":
                print("差评抓取中-------")
                path = 'jd_cn_cp.csv'
                base_url = 'https://sclub.jd.com/comment/productPageComments.action?' \
                           'callback=fetchJSON_comment98vv14056&productId={}&score={}&sortType=5&pageSize=10&page='.format(
                    li, score)
                pages = get_page({'user_agent': get_user_agent()}, base_url)

                for i in range(0, min(pages,40)):
                    if i % 15 == 0:
                        time.sleep(600)
                        url = base_url + str(i)
                        print('正在抓取第{}页'.format(i+1))
                        headers = {'user_agent': get_user_agent()}
                        try:
                            get_comments(url, headers, path)
                            time.sleep(random.randint(10, 20))
                        except:
                            print('第{}页抓取失败'.format(i+1))
                            lose.append(i)
                            print(lose)
                            continue
                    else:
                        url = base_url + str(i)
                        print('正在抓取第{}页'.format(i+1))
                        headers = {'user_agent': get_user_agent()}
                        try:
                            get_comments(url, headers, path)
                            time.sleep(random.randint(10, 20))
                        except:
                            print('第{}页抓取失败'.format(i+1))
                            lose.append(i)
                            print(lose)
                            continue
            else:
                print("好评抓取中-------")
                path = 'jd_cn_hp.csv'
                base_url = 'https://sclub.jd.com/comment/productPageComments.action?' \
                           'callback=fetchJSON_comment98vv14056&productId={}&score={}&sortType=5&pageSize=10&page='.format(
                    li, score)
                pages = get_page({'user_agent': get_user_agent()}, base_url)

                for i in range(0, min(pages, 40)):
                    if i % 15 == 0:
                        time.sleep(600)
                        url = base_url + str(i)
                        print('正在抓取第{}页'.format(i+1))
                        headers = {'user_agent': get_user_agent()}
                        try:
                            get_comments(url, headers, path)
                            time.sleep(random.randint(10, 20))
                        except:
                            print('第{}页抓取失败'.format(i+1))
                            lose.append(i)
                            print(lose)
                            continue
                    else:
                        url = base_url + str(i)
                        print('正在抓取第{}页'.format(i+1))
                        headers = {'user_agent': get_user_agent()}
                        try:
                            get_comments(url, headers, path)
                            time.sleep(random.randint(10, 20))
                        except:
                            print('第{}页抓取失败'.format(i+1))
                            lose.append(i)
                            print(lose)
                            continue



