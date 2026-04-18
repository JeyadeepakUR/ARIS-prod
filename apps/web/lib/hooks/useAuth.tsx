"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { apiLogin, apiLogout, apiRegister, type UserProfile } from "../api/auth";

type AuthContextValue = {
  user: UserProfile | null;
  isReady: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isReady, setIsReady] = useState(false);

  // Restore session from stored token by checking localStorage for a saved user.
  useEffect(() => {
    const stored = localStorage.getItem("aris_user");
    if (stored) {
      try {
        setUser(JSON.parse(stored) as UserProfile);
      } catch {
        localStorage.removeItem("aris_user");
      }
    }
    setIsReady(true);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const profile = await apiLogin(email, password);
    localStorage.setItem("aris_user", JSON.stringify(profile));
    setUser(profile);
  }, []);

  const register = useCallback(async (email: string, password: string) => {
    const profile = await apiRegister(email, password);
    localStorage.setItem("aris_user", JSON.stringify(profile));
    setUser(profile);
  }, []);

  const logout = useCallback(() => {
    apiLogout();
    localStorage.removeItem("aris_user");
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, isReady, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
