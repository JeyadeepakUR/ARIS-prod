import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";

import "./globals.css";
import { AuthProvider } from "../lib/hooks/useAuth";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  display: "swap",
});

export const metadata: Metadata = {
  title: "ARIS",
  description: "Autonomous Research Intelligence System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={spaceGrotesk.variable}>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}