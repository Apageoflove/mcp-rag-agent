"""审核探针：故意留下若干可审问题。"""


def average(values):
    return sum(values) / len(values)


def paginate(items, page, per_page=20):
    start = page * per_page
    end = start + per_page
    return items[start:end]


def parse_port(text):
    return int(text)
