from fastapi import FastAPI
from measurement import router as measurement_router
from vton_module import router as vton_router
from amazon_product_sizes import router as amazon_product_size_router

app = FastAPI(title="Unified Body Measurement & Virtual Try-On API")

app.include_router(measurement_router)
app.include_router(vton_router)
app.include_router(amazon_product_size_router)
