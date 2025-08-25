import { ThemeProvider } from '@geti/ui/theme';
import type { ReactNode } from 'react';

export const Providers = ({ children }: { children: ReactNode }) => {
  return <ThemeProvider>{children}</ThemeProvider>;
};
