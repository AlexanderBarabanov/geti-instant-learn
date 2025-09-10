/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Image } from '@geti-prompt/icons';
import { View } from '@geti/ui';

import { NoMediaPlaceholder } from '../../../../components/no-media-placeholder/no-media-placeholder.component';

const NoImagesPlaceholder = () => {
    return (
        <View minHeight={'size-3000'} height={'50%'} maxHeight={'size-5000'}>
            <NoMediaPlaceholder title={'Capture/Add images for visual prompt'} img={<Image />} />
        </View>
    );
};

export const VisualPrompt = () => {
    return <NoImagesPlaceholder />;
};
