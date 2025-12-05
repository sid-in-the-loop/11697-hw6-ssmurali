#!/usr/bin/env python3
"""
URL Configuration for Airline Policy Knowledge Collection

This file contains all the URLs organized by category for easy management.
You can easily add, remove, or modify URLs here without touching the main scripts.
"""

URLS = {
    # Baggage Policies
    'baggage': [
        # USA
        "https://www.aa.com/i18n/travel-info/baggage/checked-baggage-policy.jsp",
        "https://travelpro.com/pages/united-airlines-carry-on-bag-size-rules-restrictions",
        "https://www.terminalassistance.com/blog/alaska-airlines-baggage-allowance/",
        "https://travelpro.com/pages/jet-blue-carry-on-size-and-bag-policy",
        "https://www.mybaggage.com/shipping/airlines/american-airlines-baggage-allowance-and-fees/",
        "https://www.skyteam.com/en/flight-and-destinations/baggage",
        "https://www.skywest.com/fly-skywest-airlines/customer-information#!baggage-information",
        "https://www.allegiantair.com/popup/optional-services-fees#baggage",
        # Canada
        "https://www.aircanada.com/ca/en/aco/home/plan/baggage.html#/",
        "https://www.flyporter.com/en-ca/travel-information/baggage/carry-on-allowance",
        "https://cms.volaris.com/es/informacion-util/politicas-de-equipaje/?countryflag=Mexico&currency=MXN&Customer_ID=21&Customer_Email=WebAnonymous",
        "https://www.mybaggage.com/shipping/airlines/air-canada-baggage-allowance/",
        "https://www.cabinzero.com/blogs/air-travel-tips/air-canada-carry-on-size",
        "https://www.reddit.com/r/aircanada/comments/1h6rg7p/mod_note_the_new_air_canada_carryon_policy/",
        "https://www.skyscanner.ca/tips-and-inspiration/air-canada-baggage-rules",
        # UK, Ireland and Germany
        "https://www.mybaggage.com/shipping/airlines/british-airways/",
        "https://www.britishairways.com/content/information/baggage-essentials/liquids-and-restrictions",
        "https://www.britishairways.com/content/information/baggage-essentials",
        "https://www.sendmybag.com/airlines/british-airways-baggage-allowance/",
        "https://www.terminalassistance.com/blog/british-airways-baggage-allowance/",
        "https://stasher.com/blog/british-airways-baggage-allowance",
        "https://www.condor.com/de/flug-vorbereiten/gepaeck-tiere/zusatzgepaeck.jsp",
        "https://www.eurowings.com/en/information/baggage.html",
        "https://www.aerlingus.com/prepare/bags/checked-baggage/#/tab-0-flights-within-europe",
        "https://www.cabinzero.com/blogs/air-travel-tips/british-airways-baggage-allowance",
        "https://en.uhomes.com/blog/british-airways-baggage-size-and-weight",
        "https://www.easyjet.com/en/help/baggage/cabin-bags",
        "https://www.jet2.com/baggage",
        # Other EU
        "https://www.norwegian.com/en/travel-info/baggage/",
        "https://www.austrian.com/us/en/baggage",
        "https://www.ita-airways.com/it_it/supporto/baggage-assistance.html",
        "https://www.transavia.com/help/en-eu/baggage",
        "https://www.frenchbee.com/en/baggage/",
        "https://www.corsair.fr/en/prepare-your-trip/baggage",
        "https://www.transavia.com/en-EU/service/baggage/",
        "https://wwws.airfrance.ch/en/information/bagages",
        "https://wwws.airfrance.fr/en/information/bagages",
        "https://www.mybaggage.com/shipping/airlines/air-france-baggage-allowance/",
        "https://www.kayak.com/news/air-france-carry-on-size/",
        "https://www.s7.ru/en/help-center/category/bagazh-i-ruchnaya-klad/ruchnaya-klad/",
        "https://www.sunexpress.com/en-gb/information/luggage-info",
        # UAE and Middle East
        "https://www.etihad.com/en-us/help/baggage-information",
        "https://www.mybaggage.com/shipping/airlines/emirates-baggage-allowance/",
        "https://www.fondtravels.com/blog/travel-tips/emirates-baggage-allowance",
        "https://airadvisor.com/en/baggage-allowance/emirates-airlines",
        "https://stasher.com/blog/emirates-baggage-allowance",
        "https://www.emirates.com/us/english/before-you-fly/baggage/checked-baggage/",
        "https://www.emirates.com/us/english/before-you-fly/baggage/",
        "https://www.sendmybag.com/airlines/emirates-baggage-allowance/",
        "https://www.virginatlantic.com/gb/en/travel-information/baggage.html",
        "https://www.qatarairways.com/en/baggage.html",
        "https://www.saudia.com/pages/before-flying/baggage/baggage-allowances?sc_lang=en&sc_country=US",
        # India
        "https://www.akasaair.com/quick-links/baggage",
        "https://www.airchina.us/US/GB/info/Baggage-Restrictions/",
        # Japan and other Asian places
        "https://www.jal.co.jp/jp/en/inter/baggage/",
    ],
    
    # Fees and Charges
    'fees': [
        # USA
        "https://thriftytraveler.com/guides/airlines/united-baggage-fees/",
        "https://www.aa.com/i18n/customer-service/support/optional-service-fees.jsp",
        "https://www2.hawaiianairlines.com/legal/list-of-all-fees?clientId=126580914.1763780766&sessionId=1763780766",
        "https://www.southwest.com/html/customer-service/travel-fees.html?clk=HOME-BOOKING-WIDGET-BAGGAGE-FEES",
        "https://www.alaskaair.com/content/travel-info/optional-services-fees?lid=HomePage_Beta:BaggageFeesAndOptionalServices&int=AS_Home_MSG2_-prodID:Awareness",
        "https://www.flyfrontier.com/optional-services",
        "https://www.aveloair.com/optional-services",
        "https://www.allegiantair.com/popup/optional-services-fees",
        # Canada
        "https://www.westjet.com/en-ca/flights/fares",
        # India
        "https://corporate.spicejet.com/FeesCharges.aspx",
    ],
    
    # Ticket Changes and Cancellations
    'ticket_changes': [
        # USA
        "https://www.fondtravels.com/blog/travel-tips/american-airlines-flight-change-policy",
        "https://traveltrek.us/flights/change/delta-airlines-flight-change-policy/",
        # Japan and other Asian places
        "https://www.jal.co.jp/jp/en/dom/change/normal.html",
    ],
    
    # General Policies
    'general': [
        # USA
        "https://www.faa.gov/regulations_policies/rulemaking",
        "https://www.staralliance.com/en/benefits-and-privileges",
        "https://www.psaairlines.com/flight-attendants/",
        "https://www.travelocity.com/",
        "https://www.kalittaair.com/about",
        "https://www.mesa-air.com/passenger-information",
        "https://www.flybreeze.com/breezy-rewards-info",
        # Canada
        "https://www.vivaaerobus.com/es-mx/avisos/afectaciones-operativas",
        "https://cargojet.com/about-us/",
        "https://www.flyflair.com/travel-info",
        "https://www.volaris.com",
        "https://www.sunwing.ca/en/",
        # UK, Ireland and Germany
        "https://www.britishairways.com/travel/home/public/en_us/",
        "https://www.tui.com/flug/",
        "https://www.aerlingus.com/support/",
        "https://www.flybe.com/",
        "https://www.loganair.co.uk/customer-support/",
        "https://www.tui.co.uk/",
        "https://help.ryanair.com/hc/en-gb",
        "https://www.virginatlantic.com/en-US/help",
        # Other EU
        "https://www.flysas.com/en/support-information",
        "https://www.swiss.com/us/en/customer-support/contact-us",
        "https://www.airdolomiti.eu/support-and-contacts",
        "https://www.aireuropa.com/us/en/home?_gl=1*10lkkdh*_up*MQ..*_ga*MTIwNjIzNzQ0OS4xNzYzNzgzNTUw*_ga_3FP4QGJ6VF*czE3NjM3ODM1NDkkbzEkZzEkdDE3NjM3ODM3MjAkajYwJGwwJGgw",
        "https://help.vueling.com/hc/en-gb/categories/19798714348177-Flying-With-Kids",
        "https://support.frenchbee.com/hc/en-us?_gl=1*1oi40ww*_gcl_au*MzUyNTQ4MTc5LjE3NjM3ODM1NDE.*_ga*OTE5NTQwNTUzLjE3NjM3ODM1NDM.*_ga_L4KY30N8F3*czE3NjM3ODM1NDMkbzEkZzEkdDE3NjM3ODM5MDQkajQ1JGwwJGgxMzU4NzU2NDg2*_fplc*NlM5M3JybEYlMkZ2V2loWXVjRW5ROUJncjI4aDVBOVdkTjNxMWZza1V3QXhqVWxBSGhHY3RYUnN0ZiUyRnhTTmRxWVpTWFY5UCUyQldjYVVjbU1neU5JR3cyVnRCbSUyQldPRkFtaEN0M1FBWnF5a2ZSUUo1cWtEWUxEV2Z3MGdkeThvS1ElM0QlM0Q.",
        "https://www.aircaraibes.com/en/company-overview",
        "https://www.uralairlines.ru/en/partner-services/",
        "https://www.utair.ru/support?_gl=1*1j8t2v5*_gcl_au*MjAxMTY2MzIzMC4xNzYzNzgzNTg4*_ga*Mzk2MTEwNTU4LjE3NjM3ODM1ODc.*_ga_K2EPVEL7D5*czE3NjM3ODM1ODckbzEkZzEkdDE3NjM3ODQwMzUkajYwJGwwJGgxMzU4NzU2NDg2*_fplc*UWxNS2hmJTJGRGc0Z0V2RHp1V0JRZEhVcWh0JTJGZUs4ZTh1Und1ZlBDa3dYVnFEQTZlOWpXd0dEd0hBS3VNamJTMW5acTVEWjVleGpJZCUyQkluQlMxRUExbG04TDJJS1ltRnhHSllJcXJ2M2xrWUU2M3prMkp2azJvS0N1WjZtekd3JTNEJTNE",
        "https://www.utair.ru/support/4",
        "https://www.utair.ru/support/5",
        "https://www.utair.ru/support/3",
        "https://www.utair.ru/support/2",
        "https://www.utair.ru/support/1",
        # UAE and Middle East
        "https://www.airarabia.com/en/frequently-asked-questions",
        "https://www.flydubai.com/en-us/",
        "https://www.flynas.com/en",
        # India
        "https://www.airindia.com/in/en/frequently-asked-questions/booking.html",
        "https://www.ceair.com/",
        "https://new.hnair.com/hainanair/asmallibe/common/allchannelSeatMap.do",
        "https://www.xiamenair.com/en-us/article-detail?articleLink=%2Fcms-i18n-ow%2Fcms-en-us%2Fchannels%2F10655.json",
        "https://en.ch.com/booking-publicity",
        "https://global.sichuanair.com/www/?country=US-EN#/information?key1=1&key2=1.1",
        "https://www.flypeach.com/",
        "https://www.jetstar.com/jp/en/terms-and-conditions?pid=mainfooter:terms-and-conditions",
        # Japan and other Asian places
        "https://www.ana.co.jp/",
    ],
}

PDFS = [
    "pdfs/1.pdf",
    "pdfs/2.pdf",
    "pdfs/3.pdf",
    "pdfs/4.pdf",
    "pdfs/5.pdf",
    "pdfs/6.pdf",
    "pdfs/7.pdf",
    "pdfs/8.pdf",
    "pdfs/9.pdf",
    "pdfs/10.pdf",
    "pdfs/11.pdf",
    "pdfs/12.pdf",
    "pdfs/13.pdf",
    "pdfs/14.pdf",
    "pdfs/15.pdf",
    "pdfs/16.pdf",
    "pdfs/17.pdf",
]

def get_urls_by_category(categories=None):
    """
    Get URLs by category.
    
    Args:
        categories (list): List of category names. If None, returns all URLs.
    
    Returns:
        list: List of URLs
    """
    if categories is None:
        categories = list(URLS.keys())
    
    urls = []
    for category in categories:
        if category in URLS:
            urls.extend(URLS[category])
    
    return urls

def get_all_urls():
    """Get all URLs from all categories."""
    return get_urls_by_category()

def get_categories():
    """Get list of available categories."""
    return list(URLS.keys())

def add_urls(category, urls):
    """
    Add URLs to a category.
    
    Args:
        category (str): Category name
        urls (list): List of URLs to add
    """
    if category not in URLS:
        URLS[category] = []
    URLS[category].extend(urls)

def remove_urls(category, urls):
    """
    Remove URLs from a category.
    
    Args:
        category (str): Category name  
        urls (list): List of URLs to remove
    """
    if category in URLS:
        for url in urls:
            if url in URLS[category]:
                URLS[category].remove(url)

if __name__ == "__main__":
    # Print all available categories and URLs
    print("Available URL Categories:")
    print("=" * 30)
    for category, urls in URLS.items():
        print(f"\n{category.upper()} ({len(urls)} URLs):")
        for url in urls:
            print(f"  - {url}")
    
    print(f"\nTotal URLs: {len(get_all_urls())}")
    print(f"PDFs: {len(PDFS)}")


