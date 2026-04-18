import { apiFetch, setTokens, clearTokens } from "./client";

export type UserProfile = {
  id: string;
  email: string;
};

type AuthResponse = {
  access_token: string;
  refresh_token: string;
  user: UserProfile;
};

export async function apiRegister(email: string, password: string): Promise<UserProfile> {
  const data = await apiFetch<AuthResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  setTokens(data.access_token, data.refresh_token);
  return data.user;
}

export async function apiLogin(email: string, password: string): Promise<UserProfile> {
  const data = await apiFetch<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  setTokens(data.access_token, data.refresh_token);
  return data.user;
}

export function apiLogout() {
  clearTokens();
}
