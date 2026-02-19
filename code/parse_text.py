from bs4 import BeautifulSoup

def extract_text(html_text):
    if len(html_text) == 0:
        return ("", "")

    soup = BeautifulSoup(html_text, "lxml")
    normal_text = soup.get_text(separator=" ", strip=True)
    important_text_list = []

    if soup.title is not None:
        title_text = soup.title.get_text(separator = " ", strip = True)
        if title_text is not None:
            important_text_list.append(title_text)

    for tag in soup.find_all(["h1", "h2", "h3", "b", "strong"]):
        text = tag.get_text(separator = " ", strip = True)
        if text:
            important_text_list.append(text)

    important_text = " ".join(important_text_list)
    return normal_text, important_text

