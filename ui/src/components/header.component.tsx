/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, Header as SpectrumHeader, View } from '@geti/ui';

export const Header = () => {
    return (
        <View gridArea={'header'} backgroundColor={'gray-300'}>
            <Flex height='100%' alignItems={'center'} justifyContent={'space-between'} marginX='1rem' gap='size-200'>
                <SpectrumHeader>Geti Prompt</SpectrumHeader>
                <View>Project #1</View>
            </Flex>
        </View>
    );
};
