/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Navigate } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { MainContent } from './components/main-content.component';
import { RootLayout } from './root-layout.component';

export const routes = {
    root: path('/'),
    project: path('/projects/:projectId'),
};

const RedirectToProject = () => {
    // TODO: Include the logic to check if there is already a project ID, validate the ID or redirect to the first
    // project

    return <Navigate to={routes.project({ projectId: '1' })} replace />;
};

export const router = createBrowserRouter([
    {
        path: routes.root.pattern,
        element: <RedirectToProject />,
    },
    {
        path: routes.project.pattern,
        element: <RootLayout />,
        children: [
            {
                index: true,
                element: <MainContent />,
            },
        ],
    },
]);
