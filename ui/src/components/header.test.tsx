/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { screen } from '@testing-library/react';

import { routes } from '../router';
import { Header } from './header.component';

describe('Header', () => {
    it('renders header properly', () => {
        render(<Header />, { route: routes.project({ projectId: '1' }), path: routes.project.pattern });

        expect(screen.getByText('Geti Prompt')).toBeInTheDocument();
    });
});
