from config.database import engine
from database.models import Base

Base.metadata.create_all(
    bind=engine
)

print("数据库表创建完成")