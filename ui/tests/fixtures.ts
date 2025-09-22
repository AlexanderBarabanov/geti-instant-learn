/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createNetworkFixture, NetworkFixture } from '@msw/playwright';
import { expect, test as testBase } from '@playwright/test';

import { handlers, http } from '../src/api/utils';

interface Fixtures {
    network: NetworkFixture;
}

const test = testBase.extend<Fixtures>({
    network: createNetworkFixture({
        initialHandlers: handlers,
    }),
});

export { expect, test, http };
