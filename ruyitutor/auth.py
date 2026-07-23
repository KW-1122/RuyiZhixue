from __future__ import annotations
import secrets, time

class AuthManager:
    def __init__(self, storage):
        self.storage,self.tokens=storage,{}
    def login(self,user_id,password):
        user=self.storage.authenticate(user_id,password)
        if not user:return None
        token=secrets.token_urlsafe(32);self.tokens[token]={"user":user,"expires":time.time()+8*3600}
        return {"token":token,"user":user,"expires_in":28800}
    def verify(self,token,role=None):
        session=self.tokens.get(token)
        if not session or session["expires"]<time.time():return None
        user=session["user"]
        return user if not role or user["role"]==role else None
