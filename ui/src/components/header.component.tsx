/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, Header as SpectrumHeader, View } from '@geti/ui';

export const Header = () => {
    return (
        <View gridArea={'header'} backgroundColor={'gray-300'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <SpectrumHeader>Yolo</SpectrumHeader>
            </Flex>
        </View>
    );
};
