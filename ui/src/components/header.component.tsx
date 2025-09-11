/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, Header as SpectrumHeader, View } from '@geti/ui';
import { ProjectsListPanel } from 'src/features/projects-management/projects-list-panel.component';

export const Header = () => {
    const selectedProjectName = 'Project #1';

    return (
        <View gridArea={'header'} backgroundColor={'gray-300'}>
            <Flex height='100%' alignItems={'center'} justifyContent={'space-between'} marginX='1rem' gap='size-200'>
                <SpectrumHeader>Geti Prompt</SpectrumHeader>
                <ProjectsListPanel name={selectedProjectName} />
            </Flex>
        </View>
    );
};
