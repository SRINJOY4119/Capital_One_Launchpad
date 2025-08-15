from fastapi import APIRouter, UploadFile, File, Form
from market_inform_policy_capture import MarketInformPolicyCapture
from web_scrapper import scrape_agri_prices, scrape_policy_updates, scrape_links
from translation_tool import MultiLanguageTranslator

router = APIRouter()

market_capture_tool = MarketInformPolicyCapture()
translator = MultiLanguageTranslator()

@router.get("/api/v1/creditpolicy/comprehensive-analysis")
async def comprehensive_analysis():
    try:
        result = market_capture_tool.run_comprehensive_analysis()
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/webscrapper/agri-prices")
async def agri_prices(url: str, table_selector: str = "table"):
    try:
        data = scrape_agri_prices(url, table_selector)
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/webscrapper/policy-updates")
async def policy_updates(url: str, selector: str = ".policy-update"):
    try:
        updates = scrape_policy_updates(url, selector)
        return {"success": True, "updates": updates}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/webscrapper/links")
async def links(url: str, selector: str = "a"):
    try:
        links = scrape_links(url, selector)
        return {"success": True, "links": links}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/translate/text")
async def translate_text(text: str, source_lang: str = "auto", target_lang: str = "en"):
    try:
        result = translator.translate_robust(text, source_lang, target_lang)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/translate/batch")
async def batch_translate(texts: list, source_lang: str = "auto", target_lang: str = "en"):
    try:
        results = translator.batch_translate(texts, source_lang, target_lang)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}
